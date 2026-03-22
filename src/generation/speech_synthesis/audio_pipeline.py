"""
Audio Pipeline - Handles playback, buffering, and interruptions
Uses pygame for audio playback (works on Python 3.13+)
"""

import time
import threading
import queue
import os
import tempfile
from typing import Optional
from src.config.tts_config import TTSConfig

# Try to import pygame for audio playback
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
    print("✅ pygame audio initialized")
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not installed. Install with: pip install pygame")
except Exception as e:
    PYGAME_AVAILABLE = False
    print(f"⚠️  pygame init failed: {e}")


class AudioPipeline:
    """
    Manages audio playback with support for streaming and interruptions.
    
    Features:
    - Buffered playback for smooth audio
    - Track B can interrupt Track A mid-speech
    - Uses pygame for cross-platform compatibility
    """
    
    def __init__(self):
        """Initialize audio pipeline"""
        self.playback_enabled = TTSConfig.ENABLE_LOCAL_PLAYBACK and PYGAME_AVAILABLE
        
        # Playback queue
        self.audio_queue = queue.Queue()
        
        # Current playback control
        self.current_track = None
        self.is_playing = False
        self.interrupt_flag = threading.Event()
        
        # Playback thread
        self.playback_thread = None
        self.should_stop = threading.Event()
        
        # Stats
        self.tracks_played = 0
        self.interruptions = 0
        
        if self.playback_enabled:
            self._start_playback_thread()
            print("🔊 Audio Pipeline initialized with live playback (pygame)")
        else:
            print("📝 Audio Pipeline initialized (text-only mode)")
    
    def play_audio_stream(
        self, 
        audio_generator, 
        track_type: str = "A",
        text: str = ""
    ):
        """
        Play audio from streaming generator.
        
        Args:
            audio_generator: Generator yielding audio chunks
            track_type: "A" or "B"
            text: Original text (for fallback display)
        """
        # Check if we should interrupt current playback
        if track_type == "B" and self.is_playing and TTSConfig.ENABLE_INTERRUPTS:
            self._interrupt_current_playback()
        
        # Collect audio chunks
        audio_chunks = []
        
        try:
            for chunk in audio_generator:
                # Check if we've been interrupted
                if self.interrupt_flag.is_set():
                    print(f"   [Track {track_type} interrupted]")
                    return
                
                audio_chunks.append(chunk)
        
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
            if TTSConfig.FALLBACK_TO_TEXT:
                print(f"💬 [Track {track_type}]: {text}")
            return
        
        # Combine chunks into complete audio
        if not audio_chunks:
            return
        
        audio_data = b''.join(audio_chunks)
        
        # Queue for playback
        self.audio_queue.put({
            'audio': audio_data,
            'track_type': track_type,
            'text': text
        })
    
    def play_audio_complete(
        self,
        audio_data: bytes,
        track_type: str = "A",
        text: str = ""
    ):
        """
        Play complete audio clip.
        
        Args:
            audio_data: Complete audio as bytes
            track_type: "A" or "B"
            text: Original text (for display/fallback)
        """
        self.audio_queue.put({
            'audio': audio_data,
            'track_type': track_type,
            'text': text
        })
    
    def _interrupt_current_playback(self):
        """Interrupt currently playing audio"""
        if not self.is_playing:
            return
        
        print(f"   ⚡ Track B interrupting Track {self.current_track}!")
        
        # Stop pygame playback
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()
        
        self.interrupt_flag.set()
        self.interruptions += 1
        
        # Wait a moment for fade out
        time.sleep(TTSConfig.INTERRUPT_FADE_TIME)
        
        # Clear interrupt flag
        self.interrupt_flag.clear()
    
    def _start_playback_thread(self):
        """Start background thread for audio playback"""
        self.playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True
        )
        self.playback_thread.start()
    
    def _playback_loop(self):
        """Main playback loop (runs in background thread)"""
        while not self.should_stop.is_set():
            try:
                # Get next audio (with timeout)
                audio_item = self.audio_queue.get(timeout=0.1)
                
                # Extract data
                audio_data = audio_item['audio']
                track_type = audio_item['track_type']
                text = audio_item['text']
                
                # Mark as playing
                self.is_playing = True
                self.current_track = track_type
                
                # Display text
                print(f"\n🎤 [Track {track_type}]: {text}")
                
                # Play audio (if enabled)
                if self.playback_enabled:
                    try:
                        # Create temporary file for pygame
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                            tmp_file.write(audio_data)
                            tmp_filename = tmp_file.name
                        
                        # Load and play with pygame
                        pygame.mixer.music.load(tmp_filename)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to finish (check for interrupts)
                        while pygame.mixer.music.get_busy():
                            if self.interrupt_flag.is_set():
                                pygame.mixer.music.stop()
                                break
                            time.sleep(0.05)  # Check every 50ms
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp_filename)
                        except OSError:
                            pass
                        
                    except Exception as e:
                        print(f"❌ Playback error: {e}")
                
                # Save to file (if enabled)
                if TTSConfig.SAVE_AUDIO_FILES:
                    self._save_audio_file(audio_data, track_type)
                
                # Mark as done
                self.is_playing = False
                self.current_track = None
                self.tracks_played += 1
                
            except queue.Empty:
                # No audio in queue, continue loop
                continue
            
            except Exception as e:
                print(f"❌ Playback loop error: {e}")
                self.is_playing = False
    
    def _save_audio_file(self, audio_data: bytes, track_type: str):
        """Save audio to file for debugging"""
        os.makedirs(TTSConfig.AUDIO_OUTPUT_DIR, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        filename = f"track_{track_type}_{timestamp}.mp3"
        filepath = os.path.join(TTSConfig.AUDIO_OUTPUT_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        
        print(f"   💾 Saved: {filepath}")
    
    def wait_until_finished(self, timeout: Optional[float] = None):
        """
        Wait until all queued audio has finished playing.
        
        Args:
            timeout: Max time to wait (seconds), None = wait forever
        """
        start_time = time.time()
        
        while not self.audio_queue.empty() or self.is_playing:
            if timeout and (time.time() - start_time) > timeout:
                break
            time.sleep(0.1)
    
    def clear_queue(self):
        """Clear all pending audio"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def shutdown(self):
        """Stop playback thread and cleanup"""
        print("\n🛑 Shutting down audio pipeline...")
        
        # Stop any playing audio
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()
        
        # Stop playback thread
        self.should_stop.set()
        
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        
        # Quit pygame
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        
        print("✅ Audio pipeline stopped")
    
    def get_stats(self) -> dict:
        """Get playback statistics"""
        return {
            'tracks_played': self.tracks_played,
            'interruptions': self.interruptions,
            'queue_size': self.audio_queue.qsize(),
            'currently_playing': self.is_playing,
            'current_track': self.current_track
        }


# Quick test
if __name__ == "__main__":
    """Test audio pipeline"""
    
    print("=" * 50)
    print("AUDIO PIPELINE TEST")
    print("=" * 50)
    
    pipeline = AudioPipeline()
    
    print(f"\nPlayback enabled: {pipeline.playback_enabled}")
    print(f"Stats: {pipeline.get_stats()}")
    
    pipeline.shutdown()