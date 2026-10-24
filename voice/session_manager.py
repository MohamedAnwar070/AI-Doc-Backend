"""
Voice conversation session manager.

Each WebSocket connection creates a session that tracks:
    • Conversation history (for multi-turn context)
    • Current state (idle / listening / processing / speaking)
    • Audio buffer state
    • Cancellation signals for interrupt handling
    • Session metadata (language, user preferences)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    IDLE       = "idle"
    LISTENING  = "listening"
    PROCESSING = "processing"
    SPEAKING   = "speaking"
    ERROR      = "error"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str          # "user" | "assistant"
    text: str
    language: str = "en"
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoiceSession:
    """State container for a single voice conversation session."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: SessionState = SessionState.IDLE
    language: str = "auto"             # "auto" | "en" | "ta"
    voice: str = "af_heart"
    speed: float = 1.0

    # Conversation history — kept for multi-turn context
    history: List[ConversationTurn] = field(default_factory=list)
    max_history_turns: int = 10

    # Interrupt / cancellation
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Client confirms AudioTrack drained (optional; server falls back to duration hold)
    playback_complete_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_audio_duration_ms: int = 0

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # Session-specific document scope (optional — search across all docs by default)
    doc_id: Optional[str] = None
    category: Optional[str] = None

    def add_user_turn(self, text: str, language: str = "en") -> None:
        """Record a user utterance."""
        self.history.append(ConversationTurn(role="user", text=text, language=language))
        self._trim_history()
        self.last_activity = time.time()

    def add_assistant_turn(self, text: str) -> None:
        """Record an assistant response."""
        self.history.append(ConversationTurn(role="assistant", text=text))
        self._trim_history()
        self.last_activity = time.time()

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Return conversation history as a list of role/content dicts
        suitable for inclusion in an LLM prompt."""
        return [
            {"role": turn.role, "content": turn.text}
            for turn in self.history
        ]

    def request_interrupt(self) -> None:
        """Signal that the user wants to interrupt the current response."""
        self.cancel_event.set()
        logger.info("Session %s: interrupt requested", self.session_id)

    def clear_interrupt(self) -> None:
        """Reset the cancellation signal for a new turn."""
        self.cancel_event.clear()

    def _trim_history(self) -> None:
        """Keep only the last N turns to avoid unbounded memory growth."""
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def to_dict(self) -> Dict[str, Any]:
        """Serialise session state for WebSocket status messages."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "language": self.language,
            "voice": self.voice,
            "history_turns": len(self.history),
            "age_seconds": round(self.age_seconds, 1),
        }


# ── Global session registry ─────────────────────────────────────────────────

class SessionManager:
    """Manages all active voice sessions.

    Thread-safe via asyncio lock. Automatically cleans up stale sessions.
    """

    def __init__(self, max_session_duration: int = 1800):
        self._sessions: Dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()
        self._max_duration = max_session_duration  # seconds

    async def create_session(self, **kwargs) -> VoiceSession:
        """Create a new voice session."""
        async with self._lock:
            session = VoiceSession(**kwargs)
            self._sessions[session.session_id] = session
            logger.info(
                "Created voice session: %s  (active sessions: %d)",
                session.session_id, len(self._sessions),
            )
            return session

    async def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    async def destroy_session(self, session_id: str) -> None:
        """Remove a session and clean up its resources."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.cancel_event.set()  # Cancel any ongoing work
                logger.info(
                    "Destroyed voice session: %s  (age: %.0fs, turns: %d)",
                    session_id, session.age_seconds, len(session.history),
                )

    async def cleanup_stale(self) -> int:
        """Remove sessions that have exceeded their max duration or idle timeout.
        Returns the number of cleaned sessions."""
        async with self._lock:
            now = time.time()
            stale = [
                sid for sid, s in self._sessions.items()
                if (now - s.created_at) > self._max_duration
                or (now - s.last_activity) > 600  # 10 min idle timeout
            ]
            for sid in stale:
                session = self._sessions.pop(sid)
                session.cancel_event.set()
                logger.info("Cleaned stale session: %s", sid)
            return len(stale)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# Singleton
_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager singleton."""
    global _manager
    if _manager is None:
        from config import settings
        max_dur = getattr(settings, "voice_max_session_duration", 1800)
        _manager = SessionManager(max_session_duration=max_dur)
    return _manager
