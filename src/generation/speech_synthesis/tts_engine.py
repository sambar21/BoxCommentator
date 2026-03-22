"""
TTS Engine - ElevenLabs integration for text-to-speech with speed control
"""

import os
import time
from typing import Optional, Generator
from src.config.tts_config import TTSConfig

# Try to import ElevenLabs
try:
    from elevenlabs import ElevenLabs, VoiceSettings
    from elevenlabs.client import ElevenLabs as ElevenLabsClient
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️  elevenlabs package not installed. Install with: pip install elevenlabs")


class TTSEngine:
    """
    Text-to-Speech engine using ElevenLabs streaming API.
    
    Converts commentary text to audio in real-time with minimal latency.
    Now supports speed control!
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TTS engine.
        
        Args:
            api_key: ElevenLabs API key (defaults to ELEVEN_API_KEY env var)
        """
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("elevenlabs package required. Install: pip install elevenlabs")
        
        # Get API key
        self.api_key = api_key or os.getenv("ELEVEN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVEN_API_KEY env var or pass api_key parameter.\n"
                "Get your key at: https://elevenlabs.io/app/settings/api-keys"
            )
        
        # Initialize client
        self.client = ElevenLabsClient(api_key=self.api_key)
        
        # Voice settings with speed control
        try:
            # Try to create settings with speed parameter (newer SDK)
            self.voice_settings = VoiceSettings(
                stability=TTSConfig.STABILITY,
                similarity_boost=TTSConfig.SIMILARITY,
                style=TTSConfig.STYLE,
                use_speaker_boost=TTSConfig.USE_SPEAKER_BOOST
            )
            self.speed_supported = False
            
            # Check if speed is supported by trying to set it
            try:
                test_settings = VoiceSettings(
                    stability=TTSConfig.STABILITY,
                    similarity_boost=TTSConfig.SIMILARITY,
                    style=TTSConfig.STYLE,
                    use_speaker_boost=TTSConfig.USE_SPEAKER_BOOST,
                    speed=TTSConfig.SPEECH_SPEED
                )
                self.voice_settings = test_settings
                self.speed_supported = True
            except (TypeError, AttributeError):
                pass
                
        except Exception as e:
            print(f"⚠️  Error creating voice settings: {e}")
            self.voice_settings = VoiceSettings(
                stability=TTSConfig.STABILITY,
                similarity_boost=TTSConfig.SIMILARITY
            )
            self.speed_supported = False
        
        # Usage tracking
        self.total_characters = 0
        self.requests_made = 0
        self.failed_requests = 0
        
        # Performance metrics
        self.latencies = []
        
        print("🎤 ElevenLabs TTS Engine initialized")
        print(f"   Model: {TTSConfig.MODEL}")
        print(f"   Track A Voice: {TTSConfig.VOICE_TRACK_A}")
        print(f"   Track B Voice: {TTSConfig.VOICE_TRACK_B}")
        if self.speed_supported:
            print(f"   Speech Speed: {TTSConfig.SPEECH_SPEED}x")
        else:
            print(f"   ⚠️  Speech speed not supported - update elevenlabs package")
    
    def synthesize_streaming(
        self, 
        text: str, 
        track_type: str = "A"
    ) -> Generator[bytes, None, None]:
        """
        Generate audio from text using streaming.
        
        Args:
            text: Commentary text to synthesize
            track_type: "A" (analyst) or "B" (reactor) for voice selection
        
        Yields:
            Audio chunks as bytes
        """
        if not text or not text.strip():
            return
        
        # Get appropriate voice
        voice_id = TTSConfig.get_voice_for_track(track_type)
        
        # Track usage
        char_count = len(text)
        self.total_characters += char_count
        self.requests_made += 1
        
        # Warn if approaching free tier limit
        if TTSConfig.TRACK_CHARACTER_USAGE:
            usage_pct = (self.total_characters / TTSConfig.FREE_TIER_LIMIT) * 100
            if usage_pct >= TTSConfig.WARN_AT_PERCENTAGE:
                print(f"\n⚠️  TTS Usage Warning: {usage_pct:.1f}% of free tier used ({self.total_characters}/{TTSConfig.FREE_TIER_LIMIT} chars)")
        
        start_time = time.time()
        first_chunk_received = False
        
        try:
            # Prepare API call parameters
            api_params = {
                "text": text,
                "voice_id": voice_id,
                "model_id": TTSConfig.MODEL,
                "voice_settings": self.voice_settings,
                "optimize_streaming_latency": TTSConfig.OPTIMIZE_LATENCY,
                "output_format": TTSConfig.OUTPUT_FORMAT
            }
            
            # Generate audio stream
            audio_stream = self.client.text_to_speech.convert(**api_params)
            
            # Stream audio chunks
            for chunk in audio_stream:
                if not first_chunk_received:
                    first_chunk_time = time.time() - start_time
                    self.latencies.append(first_chunk_time)
                    first_chunk_received = True
                    
                    # Check if we're meeting latency targets
                    if first_chunk_time * 1000 > TTSConfig.FIRST_CHUNK_TIMEOUT:
                        print(f"⚠️  High latency: {first_chunk_time*1000:.0f}ms for first chunk")
                
                yield chunk
        
        except Exception as e:
            self.failed_requests += 1
            print(f"❌ TTS Error: {e}")
            
            if TTSConfig.FALLBACK_TO_TEXT:
                print(f"💬 [FALLBACK TEXT]: {text}")
            
            raise
    
    def synthesize_complete(
        self, 
        text: str, 
        track_type: str = "A"
    ) -> bytes:
        """
        Generate complete audio (non-streaming).
        Useful for short clips or when you need the full audio upfront.
        
        Args:
            text: Commentary text
            track_type: "A" or "B"
        
        Returns:
            Complete audio as bytes
        """
        audio_chunks = []
        
        for chunk in self.synthesize_streaming(text, track_type):
            audio_chunks.append(chunk)
        
        return b''.join(audio_chunks)
    
    def health_check(self) -> bool:
        """
        Check if TTS service is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get voices (quick API check)
            voices = self.client.voices.get_all()
            return len(voices.voices) > 0
        except Exception as e:
            print(f"❌ TTS Health Check Failed: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get usage and performance statistics"""
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        return {
            'total_characters': self.total_characters,
            'requests_made': self.requests_made,
            'failed_requests': self.failed_requests,
            'success_rate': (self.requests_made - self.failed_requests) / max(1, self.requests_made),
            'avg_first_chunk_latency_ms': avg_latency * 1000,
            'free_tier_usage_pct': (self.total_characters / TTSConfig.FREE_TIER_LIMIT) * 100,
            'speed_supported': self.speed_supported,
            'current_speed': TTSConfig.SPEECH_SPEED if self.speed_supported else 1.0
        }
    
    def reset_stats(self):
        """Reset usage counters (useful for testing)"""
        self.total_characters = 0
        self.requests_made = 0
        self.failed_requests = 0
        self.latencies = []


# Quick test
if __name__ == "__main__":
    """Test TTS engine"""
    
    print("=" * 50)
    print("TTS ENGINE TEST")
    print("=" * 50)
    
    try:
        engine = TTSEngine()
        
        if engine.health_check():
            print("\n✅ TTS Engine healthy!")
            
            test_text = "He's working that jab beautifully!"
            print(f"\nTest text: '{test_text}'")
            print("Generating audio...")
            
            audio = engine.synthesize_complete(test_text, track_type="A")
            
            print(f"\n✅ Generated {len(audio)} bytes of audio")
            
            # Save to file
            if TTSConfig.SAVE_AUDIO_FILES:
                os.makedirs(TTSConfig.AUDIO_OUTPUT_DIR, exist_ok=True)
                filepath = f"{TTSConfig.AUDIO_OUTPUT_DIR}/test.mp3"
                with open(filepath, 'wb') as f:
                    f.write(audio)
                print(f"💾 Saved to: {filepath}")
            
            # Show stats
            stats = engine.get_stats()
            print(f"\nStats:")
            print(f"  Characters used: {stats['total_characters']}")
            print(f"  Latency: {stats['avg_first_chunk_latency_ms']:.0f}ms")
            print(f"  Speed supported: {stats['speed_supported']}")
            if stats['speed_supported']:
                print(f"  Current speed: {stats['current_speed']}x")
        
        else:
            print("\n❌ TTS Engine not healthy")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")