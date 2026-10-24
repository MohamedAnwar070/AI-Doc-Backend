"""
WebSocket endpoint for realtime voice conversations.

Protocol:
    Client → Server:
        Binary frames : Raw Int16 PCM audio (16 kHz mono) from browser mic
        Text frames   : JSON control messages
            {"type": "start_session", "config": {"language": "auto", "doc_id": "...", "voice": "af_heart"}}
            {"type": "end_session"}
            {"type": "interrupt"}
            {"type": "config", "language": "ta", "voice": "af_heart"}
            {"type": "playback_complete"}  # optional: client finished playing TTS

    Server → Client:
        Binary frames : Raw Int16 PCM audio (24 kHz mono) for playback
        Text frames   : JSON event messages
            {"type": "session_started", "session_id": "...", "config": {...}}
            {"type": "state", "state": "listening|processing|speaking"}
            {"type": "transcript", "text": "...", "language": "en", "is_final": true}
            {"type": "response_start"}
            {"type": "response_text", "text": "...", "sentence_index": 0}
            {"type": "response_end", "full_text": "...", "sources": [...],
             "audio_duration_ms": 1234}
            {"type": "error", "message": "..."}
            {"type": "session_ended"}

Echo mitigation: after TTS bytes are sent the session stays SPEAKING until
``playback_complete`` or ``audio_duration_ms + 500`` ms (whichever first),
so the assistant does not re-ingest its own audio as user speech.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice.audio_processor import (
    AudioBuffer,
    browser_audio_to_whisper,
    float32_bytes_to_float32,
    int16_bytes_to_float32,
    WHISPER_SAMPLE_RATE,
)
from voice.session_manager import (
    SessionState,
    VoiceSession,
    get_session_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """Main WebSocket handler for voice conversations.

    Lifecycle:
        1. Client connects
        2. Client sends "start_session" → server loads session
        3. Client streams mic audio → server buffers → detects end-of-speech
        4. Server transcribes (STT) → searches docs (RAG) → generates response (LLM)
        5. Server streams TTS audio back to client
        6. Repeat from 3
        7. Client sends "end_session" or disconnects
    """
    await websocket.accept()
    logger.info("Voice WebSocket connected: %s", websocket.client)

    session: Optional[VoiceSession] = None
    audio_buffer: Optional[AudioBuffer] = None
    manager = get_session_manager()
    consecutive_interrupt_speech_chunks = 0
    processing_task: Optional[asyncio.Task] = None
    audio_chunks_received = 0
    audio_samples_received = 0

    def cancel_processing_task():
        nonlocal processing_task
        if session:
            session.request_interrupt()
        if processing_task and not processing_task.done():
            logger.info("Cancelling active processing task for session %s", session.session_id if session else "unknown")
            processing_task.cancel()
        processing_task = None

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                logger.info("Voice WebSocket disconnected cleanly")
                break

            # ── Text frame (JSON control messages) ──────────────────────
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _send_error(websocket, "Invalid JSON")
                    continue

                msg_type = data.get("type", "")

                # ── Start session ──────────────────────────────────────
                if msg_type == "start_session":
                    cancel_processing_task()
                    config = data.get("config", {})
                    session = await manager.create_session(
                        language=config.get("language", "auto"),
                        voice=config.get("voice", "af_heart"),
                        speed=config.get("speed", 1.0),
                        doc_id=config.get("doc_id"),
                        category=config.get("category"),
                        max_history_turns=config.get("max_history_turns", 10),
                    )
                    audio_buffer = AudioBuffer(
                        # How long the caller must stay quiet before we treat
                        # the utterance as finished. 1000 ms measurably cut
                        # people off at the natural pause after a question
                        # mark — "What is this document about? Please
                        # summarise..." was transcribed as only the first
                        # sentence. Humans routinely pause ~1 s mid-thought,
                        # so the gate must sit above that. The cost is +500 ms
                        # of latency on every turn, which is negligible next
                        # to the STT+LLM time behind it.
                        silence_duration_ms=1500,
                        sample_rate=WHISPER_SAMPLE_RATE,
                    )
                    await _send_json(websocket, {
                        "type": "session_started",
                        "session_id": session.session_id,
                        "config": session.to_dict(),
                    })
                    await _send_state(websocket, session, SessionState.LISTENING)
                    consecutive_interrupt_speech_chunks = 0
                    logger.info("Session started: %s", session.session_id)

                # ── End session ────────────────────────────────────────
                elif msg_type == "end_session":
                    cancel_processing_task()
                    if session:
                        await manager.destroy_session(session.session_id)
                    await _send_json(websocket, {"type": "session_ended"})
                    logger.info("Session ended by client")
                    break

                # ── Interrupt ──────────────────────────────────────────
                elif msg_type == "interrupt":
                    if session:
                        cancel_processing_task()
                        session.cancel_event.set()
                        try:
                            session.playback_complete_event.set()
                        except Exception:
                            pass
                        session.state = SessionState.LISTENING
                        if audio_buffer:
                            audio_buffer.reset()
                        await _send_state(websocket, session, SessionState.LISTENING)
                        consecutive_interrupt_speech_chunks = 0
                        logger.info("Interrupt: session %s", session.session_id)

                # ── Client finished playing TTS (optional) ─────────────
                elif msg_type == "playback_complete":
                    if session:
                        session.playback_complete_event.set()
                        logger.debug(
                            "playback_complete from client session=%s",
                            session.session_id,
                        )

                # ── Update config ──────────────────────────────────────
                elif msg_type == "config":
                    if session:
                        if "language" in data:
                            session.language = data["language"]
                        if "voice" in data:
                            session.voice = data["voice"]
                        if "speed" in data:
                            session.speed = float(data["speed"])
                        if "doc_id" in data:
                            session.doc_id = data.get("doc_id")
                        if "category" in data:
                            session.category = data.get("category")
                        await _send_json(websocket, {
                            "type": "config_updated",
                            "config": session.to_dict(),
                        })

                else:
                    await _send_error(websocket, f"Unknown message type: {msg_type}")

            # ── Binary frame (audio data) ───────────────────────────────
            elif "bytes" in message:
                audio_chunks_received += 1
                if session is None:
                    await _send_error(websocket, "No active session. Send start_session first.")
                    continue

                if session.state == SessionState.PROCESSING:
                    # Ignore audio while processing (unless it's an interrupt)
                    continue

                raw_bytes = message["bytes"]
                if len(raw_bytes) < 2:
                    continue

                # Convert browser PCM to Whisper format
                try:
                    # Browser sends Int16 PCM at 16kHz (downsampled in browser)
                    audio_chunk = int16_bytes_to_float32(raw_bytes)
                    audio_samples_received += len(audio_chunk)
                except Exception as e:
                    logger.warning("Audio decode error: %s", e)
                    continue

                if audio_chunks_received == 1:
                    from voice.audio_processor import compute_db
                    logger.info(
                        "First microphone audio received: %d samples, level %.1f dB",
                        len(audio_chunk),
                        compute_db(audio_chunk),
                    )

                # ── While the assistant is speaking ────────────────────
                # The mic is hearing our own TTS coming back out of the
                # phone speaker. That echo must NEVER reach the audio
                # buffer: it would both fabricate utterances and poison the
                # VAD's noise-floor estimate (which is what previously made
                # sessions hang in "listening" forever). Barge-in detection
                # is the only thing this audio is allowed to drive.
                if session.state == SessionState.SPEAKING:
                    from voice.audio_processor import is_speech
                    if is_speech(audio_chunk, threshold_db=-26.0):
                        consecutive_interrupt_speech_chunks += 1
                        if consecutive_interrupt_speech_chunks >= 2:
                            logger.info("Active user speech detected during assistant playback — triggering interrupt")
                            cancel_processing_task()
                            if audio_buffer:
                                audio_buffer.reset()
                            await _send_state(websocket, session, SessionState.LISTENING)
                            consecutive_interrupt_speech_chunks = 0
                    else:
                        consecutive_interrupt_speech_chunks = max(0, consecutive_interrupt_speech_chunks - 1)
                    # Discard the chunk either way. On a confirmed interrupt
                    # the state is now LISTENING, so the *next* chunk starts
                    # the new utterance cleanly.
                    continue

                consecutive_interrupt_speech_chunks = 0

                # Update state to listening if idle
                if session.state == SessionState.IDLE:
                    session.state = SessionState.LISTENING

                # Add to audio buffer
                if audio_buffer is None:
                    audio_buffer = AudioBuffer(sample_rate=WHISPER_SAMPLE_RATE)

                complete_utterance = audio_buffer.add_chunk(audio_chunk)

                # End-of-utterance detected — process it
                if complete_utterance is not None and len(complete_utterance) > 0:
                    cancel_processing_task()
                    processing_task = asyncio.create_task(
                        _process_utterance(
                            websocket=websocket,
                            session=session,
                            audio=complete_utterance,
                            audio_buffer=audio_buffer,
                        )
                    )

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected: %s", websocket.client)
    except Exception as e:
        logger.error("Voice WebSocket error: %s", e, exc_info=True)
        try:
            await _send_error(websocket, f"Server error: {type(e).__name__}")
        except Exception:
            pass
    finally:
        if session:
            logger.info(
                "Voice session audio summary: %d chunks, %.1f seconds received",
                audio_chunks_received,
                audio_samples_received / float(WHISPER_SAMPLE_RATE),
            )
        cancel_processing_task()
        # Cleanup session
        if session:
            await manager.destroy_session(session.session_id)
        logger.info("Voice WebSocket handler exited")


# ── Processing pipeline ─────────────────────────────────────────────────────

async def _process_utterance(
    websocket: WebSocket,
    session: VoiceSession,
    audio: "np.ndarray",
    audio_buffer: AudioBuffer,
):
    """Full pipeline: STT → RAG → LLM → TTS → stream audio back.

    This is called when an end-of-utterance is detected (silence after speech).
    """
    import numpy as np

    # Abort immediately if the WebSocket is already dead
    if not _ws_is_open(websocket):
        logger.info("WebSocket already closed before processing — aborting utterance")
        return

    # Reset interrupt flag for this new turn
    session.clear_interrupt()

    # ── 1. Speech-to-Text ────────────────────────────────────────────
    await _send_state(websocket, session, SessionState.PROCESSING)

    # Start a background keep-alive task to prevent reverse proxy/browser idle timeouts during CPU-heavy tasks
    async def keep_alive_worker():
        try:
            while True:
                await asyncio.sleep(5.0)
                # Send a processing state ping to keep the websocket active
                await _send_json(websocket, {"type": "state", "state": "processing", "keep_alive": True})
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    keep_alive_task = asyncio.create_task(keep_alive_worker())

    try:
        try:
            from voice.stt_service import transcribe

            # Determine language setting
            lang = session.language if session.language != "auto" else None

            full_text, detected_lang, segments = await transcribe(
                audio=audio,
                language=lang,
            )
        except Exception as e:
            logger.error("STT failed: %s", e, exc_info=True)
            await _send_error(websocket, f"Speech recognition failed: {type(e).__name__}")
            await _send_state(websocket, session, SessionState.LISTENING)
            return

        # Abort if WebSocket died during the slow STT step
        if not _ws_is_open(websocket):
            logger.info("WebSocket closed during STT — aborting")
            return

        if not full_text.strip():
            logger.info("STT returned empty text — resuming listening")
            await _send_state(websocket, session, SessionState.LISTENING)
            return

        # Filter out common Whisper hallucination artifacts on silent/noisy inputs
        cleaned_text = full_text.strip().lower().replace(".", "").replace(",", "").strip()
        hallucination_phrases = {
            "thank you", "thank you for watching", "thank you very much",
            "subtitles by", "amaraorg", "bye", "you", "uh", "um", "oh",
            "tamil", "english"
        }
        if cleaned_text in hallucination_phrases or len(cleaned_text) <= 2:
            logger.info("STT returned empty or hallucinated text ('%s') — resuming listening", full_text)
            await _send_state(websocket, session, SessionState.LISTENING)
            return

        # Send transcript to client
        await _send_json(websocket, {
            "type": "transcript",
            "text": full_text,
            "language": detected_lang,
            "is_final": True,
            "segments": segments,
        })

        # Record user turn
        session.add_user_turn(full_text, language=detected_lang)

        # Check for interrupt
        if session.cancel_event.is_set():
            await _send_state(websocket, session, SessionState.LISTENING)
            return

        # Abort if WebSocket died during processing
        if not _ws_is_open(websocket):
            logger.info("WebSocket closed before LLM — aborting")
            return

        # ── 2. RAG + LLM Response ────────────────────────────────────────
        response_sources = []
        try:
            from voice.conversation_engine import VoiceConversationEngine

            engine = VoiceConversationEngine()
            response_result = await engine.generate_response_with_sources(
                user_text=full_text,
                conversation_history=session.get_context_messages(),
                doc_id=session.doc_id,
                category=session.category,
                detected_language=detected_lang,
            )
            response_text = response_result.text
            response_sources = response_result.sources
        except Exception as e:
            logger.error("LLM response failed: %s", e, exc_info=True)
            response_text = "Sorry, I had trouble generating a response."

        if not response_text.strip():
            response_text = "I couldn't find relevant information for your question."

        # Check for interrupt before TTS
        if session.cancel_event.is_set():
            await _send_state(websocket, session, SessionState.LISTENING)
            return

        # Record assistant turn
        session.add_assistant_turn(response_text)

        # ── 3. Text-to-Speech + Stream ───────────────────────────────────
        await _send_json(websocket, {"type": "response_start"})
        await _send_state(websocket, session, SessionState.SPEAKING)

    finally:
        # ALWAYS cancel the keep_alive task when entering TTS speaking phase or exiting
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            pass

    total_pcm_bytes = 0
    tts_sample_rate = 24000
    try:
        from voice.tts_service import generate_speech

        sentence_idx = 0
        audio_chunks_sent = 0
        async for sentence_text, pcm_bytes in generate_speech(
            text=response_text,
            voice=session.voice,
            speed=session.speed,
            cancel_event=session.cancel_event,
        ):
            # Check interrupt
            if session.cancel_event.is_set():
                logger.info("TTS streaming interrupted")
                break

            # Send the sentence text
            await _send_json(websocket, {
                "type": "response_text",
                "text": sentence_text or response_text,
                "sentence_index": sentence_idx,
                "has_audio": bool(pcm_bytes),
                "sample_rate": tts_sample_rate,
            })

            # Send audio as binary frame. Empty chunks are ignored by the
            # browser and should not count as spoken output.
            if pcm_bytes:
                await websocket.send_bytes(pcm_bytes)
                audio_chunks_sent += 1
                total_pcm_bytes += len(pcm_bytes)
            sentence_idx += 1

        if sentence_idx == 0:
            logger.warning("TTS produced no chunks; sending text-only response")
            await _send_json(websocket, {
                "type": "response_text",
                "text": response_text,
                "sentence_index": 0,
                "has_audio": False,
                "sample_rate": tts_sample_rate,
            })
        elif audio_chunks_sent == 0 and not session.cancel_event.is_set():
            logger.warning("TTS produced text chunks but no audio bytes")

    except Exception as e:
        logger.error("TTS streaming failed: %s", e, exc_info=True)
        await _send_json(websocket, {
            "type": "response_text",
            "text": response_text,
            "sentence_index": 0,
            "has_audio": False,
            "sample_rate": 24000,
        })

    # ── 4. Done (stay SPEAKING until playback likely finished) ───────
    # Int16 mono @ sample_rate → duration_ms = bytes / 2 / rate * 1000
    audio_duration_ms = 0
    if total_pcm_bytes > 0:
        audio_duration_ms = int(total_pcm_bytes / 2 / max(tts_sample_rate, 1) * 1000)
    session.last_audio_duration_ms = audio_duration_ms
    session.playback_complete_event = asyncio.Event()

    await _send_json(websocket, {
        "type": "response_end",
        "full_text": response_text,
        "sources": response_sources,
        "audio_duration_ms": audio_duration_ms,
    })

    if session.cancel_event.is_set():
        session.clear_interrupt()
        await _send_state(websocket, session, SessionState.LISTENING)
        return

    # Hold SPEAKING while client plays buffered TTS (echo guard).
    # Prefer client playback_complete; fall back to duration + 500 ms.
    hold_ms = audio_duration_ms + 500 if audio_duration_ms > 0 else 0
    if hold_ms > 0:
        logger.info(
            "stage=tts_hold session=%s audio_duration_ms=%d hold_ms=%d",
            session.session_id,
            audio_duration_ms,
            hold_ms,
        )
        try:
            await asyncio.wait_for(
                session.playback_complete_event.wait(),
                timeout=hold_ms / 1000.0,
            )
            logger.debug("playback_complete acknowledged session=%s", session.session_id)
        except asyncio.TimeoutError:
            logger.debug(
                "playback hold timed out session=%s hold_ms=%d",
                session.session_id,
                hold_ms,
            )
        except asyncio.CancelledError:
            raise

        # Barge-in during hold
        if session.cancel_event.is_set():
            session.clear_interrupt()

    if audio_buffer:
        audio_buffer.reset()
    await _send_state(websocket, session, SessionState.LISTENING)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ws_is_open(websocket: WebSocket) -> bool:
    """Check if the WebSocket connection is still open."""
    try:
        return websocket.client_state.name == "CONNECTED"
    except Exception:
        return False


async def _send_json(websocket: WebSocket, data: dict) -> None:
    """Send a JSON message to the client. Silently skips if disconnected."""
    if not _ws_is_open(websocket):
        return
    try:
        await websocket.send_text(json.dumps(data))
    except Exception as e:
        logger.debug("Failed to send JSON (client likely disconnected): %s", e)


async def _send_state(websocket: WebSocket, session: VoiceSession, state: SessionState) -> None:
    """Update session state and notify the client."""
    session.state = state
    session.last_activity = time.time()
    await _send_json(websocket, {
        "type": "state",
        "state": state.value,
    })


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send an error message to the client."""
    logger.warning("Sending error to client: %s", message)
    await _send_json(websocket, {
        "type": "error",
        "message": message,
    })
