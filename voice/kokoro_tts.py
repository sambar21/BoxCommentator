"""
Kokoro text-to-speech for live commentary, with time-to-first-audio measurement.

Kokoro (82M parameters, open weights) runs locally on CPU or GPU. Setup:
    pip install "kokoro>=0.9.4" soundfile numpy
    plus the espeak-ng system package (apt install espeak-ng, or the Windows installer)

KokoroEngine.synthesize(text) yields float32 mono audio chunks at 24 kHz, one per sentence/segment,
so playback can start before the whole line is rendered. Any object with the same two members
(`sample_rate`, `synthesize`) can stand in, which is how the timing code is tested without a model.

Timing definitions (all milliseconds, wall clock):
    tts_first_audio_ms  text handed to the engine -> first audio chunk produced
    llm_ms              commentary text complete (the LLM stream ended)
    e2e_first_audio_ms  llm_ms + tts_first_audio_ms: event commentated -> something audible
Commentary lines are one short sentence, so there is no earlier sentence boundary to start speaking at;
e2e is therefore the LLM total time plus the TTS first-chunk time.
"""

import io
import re
import time
import wave
from dataclasses import dataclass, asdict
from typing import Iterable, Iterator, List, Optional

SAMPLE_RATE = 24000
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class KokoroEngine:
    def __init__(self, lang_code: str = "a", voice: str = "af_heart", speed: float = 1.0):
        self.lang_code, self.voice, self.speed = lang_code, voice, speed
        self.sample_rate = SAMPLE_RATE
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise RuntimeError('Kokoro is not installed: pip install "kokoro>=0.9.4" soundfile '
                                   'and install the espeak-ng system package') from e
            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def warmup(self):
        """Load the model and render a throwaway line so the first real request is not cold."""
        for _ in self.synthesize("Warming up."):
            pass

    def synthesize(self, text: str) -> Iterator["np.ndarray"]:
        pipeline = self._load()
        for _graphemes, _phonemes, audio in pipeline(text, voice=self.voice, speed=self.speed):
            if audio is None:
                continue
            yield audio.numpy() if hasattr(audio, "numpy") else audio


def split_sentences(text: str) -> List[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def to_wav_bytes(samples, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 [-1, 1] mono samples -> 16-bit PCM WAV bytes."""
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@dataclass
class SpeechTiming:
    text: str
    llm_ms: Optional[float]
    tts_first_audio_ms: Optional[float]
    tts_total_ms: float
    audio_seconds: float
    chunks: int

    @property
    def e2e_first_audio_ms(self) -> Optional[float]:
        if self.tts_first_audio_ms is None:
            return None
        return (self.llm_ms or 0.0) + self.tts_first_audio_ms

    @property
    def real_time_factor(self) -> Optional[float]:
        """Render time / audio length. Below 1.0 means faster than real time."""
        return (self.tts_total_ms / 1000.0) / self.audio_seconds if self.audio_seconds else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["e2e_first_audio_ms"] = self.e2e_first_audio_ms
        d["real_time_factor"] = self.real_time_factor
        return d


def speak(text: str, engine, llm_ms: Optional[float] = None, on_audio=None) -> SpeechTiming:
    """
    Synthesize `text`, timing the first chunk. `on_audio(chunk)` is called per chunk as it is
    produced (hand it to a player to start playback immediately).
    """
    t0 = time.perf_counter()
    first = None
    chunks = 0
    samples = 0
    for chunk in engine.synthesize(text):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        chunks += 1
        samples += len(chunk)
        if on_audio:
            on_audio(chunk)
    total = (time.perf_counter() - t0) * 1000
    return SpeechTiming(text=text, llm_ms=llm_ms, tts_first_audio_ms=first, tts_total_ms=total,
                        audio_seconds=samples / engine.sample_rate, chunks=chunks)


class Player:
    """Plays audio chunks through pygame if it is available; silent no-op otherwise."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.enabled = False
        try:
            import pygame
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
            self._pygame = pygame
            self.enabled = True
        except Exception:
            self._pygame = None

    def play(self, chunk):
        if not self.enabled:
            return
        import numpy as np
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        self._pygame.mixer.Sound(buffer=pcm).play()

    def wait(self):
        while self.enabled and self._pygame.mixer.get_busy():
            time.sleep(0.05)
