"""
Speech Synthesis Module - Convert text commentary to voice
"""

from .tts_engine import TTSEngine
from .audio_pipeline import AudioPipeline

__all__ = ['TTSEngine', 'AudioPipeline']
