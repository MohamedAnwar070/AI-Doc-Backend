"""Regression tests for the AudioBuffer end-of-speech gate.

These lock in the fix for the "session hangs in listening forever" bug, whose
root cause was a noise-floor estimate that learned from speech and from
assistant TTS echo. Because speech is always louder than silence, that made
the floor a one-way ratchet: it climbed until it crossed the speech threshold,
after which background noise scored as speech, silence never accumulated, and
end-of-utterance stopped firing.
"""

import numpy as np
import pytest

from voice.audio_processor import AudioBuffer, WHISPER_SAMPLE_RATE

SR = WHISPER_SAMPLE_RATE
CHUNK = 1600  # 100 ms — matches the Android client's frame size

# Levels measured from real phone-mic traffic in production logs.
SPEECH_DB = -26.0
LOUD_SPEECH_DB = -16.0
QUIET_SPEECH_DB = -32.0
SILENCE_DB = -55.0
ECHO_DB = -30.0  # assistant TTS returning through the speaker


def chunk_at(db: float, n: int = CHUNK, seed: int = 0) -> np.ndarray:
    """Broadband (speech-like) noise chunk at a target RMS level."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    x /= np.sqrt(np.mean(x**2))
    return (x * (10 ** (db / 20.0))).astype(np.float32)


def speak(buf: AudioBuffer, db: float, ms: int) -> list:
    return [buf.add_chunk(chunk_at(db, seed=i)) for i in range(ms // 100)]


def go_silent(buf: AudioBuffer, ms: int = 1500) -> bool:
    """Feed silence; return True if end-of-utterance fired."""
    for i in range(ms // 100):
        if buf.add_chunk(chunk_at(SILENCE_DB, seed=100 + i)) is not None:
            return True
    return False


class TestEndOfSpeech:
    def test_detects_end_of_speech_after_utterance(self):
        buf = AudioBuffer(sample_rate=SR)
        speak(buf, SPEECH_DB, ms=1500)
        assert go_silent(buf), "end-of-utterance must fire after speech + silence"

    @pytest.mark.parametrize("db", [LOUD_SPEECH_DB, SPEECH_DB, QUIET_SPEECH_DB])
    def test_detects_across_realistic_speech_levels(self, db):
        buf = AudioBuffer(sample_rate=SR)
        speak(buf, db, ms=1500)
        assert go_silent(buf), f"speech at {db} dB must be detected"

    def test_fires_on_every_turn_of_a_long_session(self):
        """The regression: turn N must behave exactly like turn 1."""
        buf = AudioBuffer(sample_rate=SR)
        for turn in range(1, 11):
            speak(buf, SPEECH_DB, ms=1500)
            assert go_silent(buf), f"end-of-utterance stopped firing at turn {turn}"

    def test_assistant_echo_does_not_break_subsequent_turns(self):
        """Echo reaching the buffer must not disable the gate.

        ws_handler now drops echo before it ever gets here, but the gate must
        survive it regardless — defence in depth.
        """
        buf = AudioBuffer(sample_rate=SR)
        for turn in range(1, 6):
            speak(buf, SPEECH_DB, ms=1500)
            assert go_silent(buf), f"end-of-utterance stopped firing at turn {turn}"
            speak(buf, ECHO_DB, ms=2000)  # assistant TTS echo
            buf.reset()


class TestNoiseFloor:
    def test_floor_never_learns_from_speech(self):
        """The ratchet guard: sustained speech must not raise the floor."""
        buf = AudioBuffer(sample_rate=SR)
        go_silent(buf, ms=1000)  # converge on the real floor
        settled = buf.noise_floor_db
        speak(buf, LOUD_SPEECH_DB, ms=3000)
        assert buf.noise_floor_db == pytest.approx(settled, abs=0.01), (
            "speech leaked into the noise-floor estimate"
        )

    def test_floor_never_learns_from_echo(self):
        buf = AudioBuffer(sample_rate=SR)
        go_silent(buf, ms=1000)
        settled = buf.noise_floor_db
        speak(buf, ECHO_DB, ms=3000)
        assert buf.noise_floor_db == pytest.approx(settled, abs=0.01)

    def test_threshold_stays_below_quiet_speech(self):
        """The gate must never demand speech louder than we expect to hear."""
        buf = AudioBuffer(sample_rate=SR)
        for _ in range(50):
            buf.add_chunk(chunk_at(SPEECH_DB))
            buf.add_chunk(chunk_at(ECHO_DB))
        assert buf.speech_threshold_db < QUIET_SPEECH_DB, (
            "threshold rose above quiet speech — the session would hang listening"
        )

    def test_floor_recovers_quickly_when_room_gets_quiet(self):
        buf = AudioBuffer(sample_rate=SR)
        buf.add_chunk(chunk_at(-36.0))  # seeded high (first chunk was noisy)
        go_silent(buf, ms=2000)
        assert buf.noise_floor_db < -50.0, "floor failed to track down to a quiet room"


class TestFalsePositives:
    def test_silence_alone_never_fires(self):
        buf = AudioBuffer(sample_rate=SR)
        assert not go_silent(buf, ms=5000)

    def test_isolated_transient_does_not_fabricate_an_utterance(self):
        """A door slam / keyboard click must decay, not accumulate."""
        buf = AudioBuffer(sample_rate=SR)
        for i in range(30):
            go_silent(buf, ms=500)
            buf.add_chunk(chunk_at(LOUD_SPEECH_DB, seed=i))  # 100 ms transient
        assert not buf._speech_detected, "transients accumulated into phantom speech"

    def test_idle_listening_does_not_grow_the_buffer(self):
        """Bounded memory: idle silence must not accumulate 30 s of audio."""
        buf = AudioBuffer(sample_rate=SR)
        go_silent(buf, ms=20_000)
        assert buf.duration_s <= 0.5, "idle buffer grew beyond the pre-roll"

    def test_preroll_preserves_speech_onset(self):
        """Some lead-in is kept so the first phoneme isn't clipped."""
        buf = AudioBuffer(sample_rate=SR)
        go_silent(buf, ms=5000)
        speak(buf, SPEECH_DB, ms=500)
        assert buf.duration_s > 0.5, "no pre-roll retained ahead of speech"
