"""
Voice Conversation Layer — Realtime voice interaction for the RAG system.

This package provides:
    • STT (Speech-to-Text) via Faster-Whisper large-v3
    • TTS (Text-to-Speech) via Kokoro
    • WebSocket-based streaming audio pipeline
    • Session management with interrupt handling
    • Conversation engine integrating existing RAG + LLM
"""

from .ws_handler import router as voice_router

__all__ = ["voice_router"]
