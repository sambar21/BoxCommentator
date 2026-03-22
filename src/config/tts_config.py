"""
TTS Configuration - ElevenLabs Voice Synthesis
"""

class TTSConfig:
    """
    Configuration for Text-to-Speech synthesis.
    Uses ElevenLabs for high-quality, low-latency streaming.
    """
    
    # ========================================
    # MAIN SWITCH
    # ========================================
    ENABLED = True  # Set False to disable TTS (text-only mode)
    
    # ========================================
    # ELEVENLABS CONFIG
    # ========================================
    PROVIDER = "elevenlabs"  # Currently only ElevenLabs supported
    
    # Voice IDs (get from: https://elevenlabs.io/app/voice-library)
    # These are example IDs - replace with your chosen voices
    VOICE_TRACK_A = "pqHfZKP75CvOlQylNhV4"  # Your calm voice
    VOICE_TRACK_B = "SOYHLrjzK2X1ezoPC6cr"  # Your excited voice
    
    # You can use the same voice for both tracks if you want
    USE_SINGLE_VOICE = True  # Set True to use VOICE_TRACK_A for everything
    
    # Model selection
    MODEL = "eleven_turbo_v2_5"  # Options: eleven_turbo_v2_5, eleven_multilingual_v2
    
    # Voice settings (0.0 to 1.0)
    STABILITY = 0.5      # Lower = more variable/emotional, Higher = more stable
    SIMILARITY = 0.75    # How closely to match the voice
    STYLE = 0.0          # 0 = ignore, >0 = exaggerate style (only for v2 models)
    USE_SPEAKER_BOOST = True  # Boost clarity
    
    # Speech speed
    SPEECH_SPEED = 1.2  # 1.0 = normal, 1.3 = 30% faster, max ~1.5
    
    # Streaming settings
    CHUNK_SIZE = 1024    # Audio chunk size for streaming
    OPTIMIZE_LATENCY = 4  # 0-4, higher = lower latency but may affect quality
    
    # ========================================
    # AUDIO OUTPUT SETTINGS
    # ========================================
    OUTPUT_FORMAT = "mp3_44100_128"  # Options: mp3_44100_128, pcm_16000, pcm_22050, pcm_24000, pcm_44100
    
    # Playback (if using local audio output)
    ENABLE_LOCAL_PLAYBACK = True  # Play audio through speakers
    
    # WebSocket (for browser streaming)
    ENABLE_WEBSOCKET_STREAM = False  # Set True to stream to browser
    WEBSOCKET_PORT = 8765
    
    # File saving (for debugging)
    SAVE_AUDIO_FILES = False  # Set True to save audio to disk
    AUDIO_OUTPUT_DIR = "./audio_output"
    
    # ========================================
    # PERFORMANCE SETTINGS
    # ========================================
    # Max time to wait for first audio chunk (ms)
    FIRST_CHUNK_TIMEOUT = 500
    
    # Buffer size for smooth playback (number of chunks)
    PLAYBACK_BUFFER_SIZE = 3
    
    # Character limit tracking (for free tier)
    TRACK_CHARACTER_USAGE = True
    FREE_TIER_LIMIT = 10000  # chars/month
    WARN_AT_PERCENTAGE = 80  # Warn at 80% usage
    
    # ========================================
    # INTERRUPTION HANDLING
    # ========================================
    # Allow Track B to interrupt Track A mid-speech
    ENABLE_INTERRUPTS = True
    
    # Fade out time when interrupting (seconds)
    INTERRUPT_FADE_TIME = 0.1
    
    # ========================================
    # FALLBACK BEHAVIOR
    # ========================================
    # What to do if TTS fails
    FALLBACK_TO_TEXT = True  # Print text if voice fails
    
    # Retry settings
    MAX_RETRIES = 2
    RETRY_DELAY = 0.5  # seconds
    
    @classmethod
    def get_voice_for_track(cls, track_type: str) -> str:
        """Get appropriate voice ID for track type"""
        if cls.USE_SINGLE_VOICE:
            return cls.VOICE_TRACK_A
        
        return cls.VOICE_TRACK_A if track_type == "A" else cls.VOICE_TRACK_B
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if TTS is enabled"""
        return cls.ENABLED