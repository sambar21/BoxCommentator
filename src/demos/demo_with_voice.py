"""
Demo: AI Commentary with VOICE OUTPUT
Shows complete integration: Trackers → Priority Queue → Consumer → LLM → TTS → Audio
"""

import time

from dotenv import load_dotenv
load_dotenv()

from src.core.action_buffer.buffer import Punch
from src.core.orchestrator import CommentaryOrchestrator
from src.generation.llm_interface.groq_client import GroqClient
from src.config.tts_config import TTSConfig

# Import TTS components (with graceful fallback)
try:
    from src.generation.speech_synthesis.tts_engine import TTSEngine
    from src.generation.speech_synthesis.audio_pipeline import AudioPipeline
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TTS not available: {e}")
    TTS_AVAILABLE = False


def simulate_fight_with_voice():
    """
    Simulate a fight with AI commentary AND voice output.
    NEW ARCHITECTURE - uses integrated orchestrator with queue consumer and dual-track generation.
    """
    print("=" * 70)
    print("🥊 AI COMMENTARY DEMO - WITH VOICE OUTPUT")
    print("=" * 70)
    
    print(f"\n📡 LLM Provider: GROQ")
    print(f"🎤 TTS Enabled: {TTSConfig.is_enabled() and TTS_AVAILABLE}")
    print()
    
    # Initialize LLM client
    print("🔧 Initializing LLM client...")
    llm_client = GroqClient()
    
    use_llm = llm_client.health_check()
    if not use_llm:
        print("⚠️  LLM not healthy - will use template messages")
    else:
        print("✅ LLM ready!")
    
    # Initialize orchestrator (now handles everything internally)
    print("🔧 Initializing orchestrator...")
    orchestrator = CommentaryOrchestrator(llm_client=llm_client)
    orchestrator.set_fighter_names("Alvarez", "Garcia")
    print("✅ Orchestrator ready!")
    
    # Create TTS components
    tts_engine = None
    audio_pipeline = None
    use_tts = False
    
    if TTSConfig.is_enabled() and TTS_AVAILABLE:
        try:
            print("\n🔧 Initializing TTS...")
            tts_engine = TTSEngine()
            audio_pipeline = AudioPipeline()
            
            if tts_engine.health_check():
                print("✅ TTS ready!")
                use_tts = True
            else:
                print("⚠️  TTS not healthy - text-only mode")
        
        except Exception as e:
            print(f"⚠️  TTS initialization failed: {e}")
            print("   Continuing in text-only mode")
    
    # Start Round 1
    print("\n" + "=" * 70)
    print("🔔 ROUND 1")
    print("=" * 70 + "\n")
    
    orchestrator.start_round(1)
    
    # Simulate punches
    current_time = time.time()
    
    punches = [
        # Opening exchange - moderate activity
        Punch(1, "jab", "head", "landed", current_time, 5),
        Punch(2, "jab", "head", "missed", current_time + 0.5, 0),
        Punch(1, "jab", "head", "landed", current_time + 1, 5),
        Punch(1, "cross", "head", "landed", current_time + 1.5, 10),
        
        # P1 combo - building excitement
        Punch(1, "hook", "body", "landed", current_time + 2, 12),
        Punch(1, "hook", "head", "landed", current_time + 2.5, 15),
        Punch(1, "uppercut", "head", "landed", current_time + 3, 20),
        
        # P2 response
        Punch(2, "cross", "head", "blocked", current_time + 4, 0),
        Punch(2, "hook", "body", "landed", current_time + 4.5, 10),
        
        # More P1 pressure - dominance shift
        Punch(1, "jab", "head", "landed", current_time + 5, 5),
        Punch(1, "jab", "head", "landed", current_time + 5.5, 5),
        Punch(1, "cross", "body", "landed", current_time + 6, 12),
        Punch(1, "hook", "body", "landed", current_time + 6.5, 15),
        
        # URGENT EVENT - Big hit (should trigger Track B)
        Punch(1, "cross", "head", "landed", current_time + 7, 35),
        
        # Lull period - should trigger idle commentary after 7 seconds
        Punch(2, "jab", "head", "missed", current_time + 15, 0),
        
        # Another exchange
        Punch(1, "jab", "head", "landed", current_time + 16, 5),
        Punch(2, "cross", "head", "landed", current_time + 16.5, 12),
        Punch(2, "hook", "body", "landed", current_time + 17, 15),
        
        # P2 momentum shift
        Punch(2, "jab", "head", "landed", current_time + 18, 5),
        Punch(2, "cross", "head", "landed", current_time + 18.5, 12),
        Punch(2, "hook", "head", "landed", current_time + 19, 18),
    ]
    
    commentary_count = 0
    
    # Process punches through NEW orchestrator
    for i, punch in enumerate(punches):
        print(f"\n⚡ Punch {i+1}: {punch.attacker} throws {punch.punch_type} to {punch.target} ({punch.outcome}, dmg={punch.damage})")
        
        # Process punch - orchestrator handles everything internally
        commentary = orchestrator.process_punch(punch)
        
        if commentary:
            commentary_count += 1
            
            # Get stats to determine track
            stats = orchestrator.get_stats()
            track_a_total = stats.get('track_a_generated', 0)
            track_b_total = stats.get('track_b_generated', 0)
            
            # Determine which track was just used
            if track_b_total > 0 and commentary_count == track_a_total + track_b_total:
                track_type = "B"
            else:
                track_type = "A"
            
            print(f"\n{'='*70}")
            print(f"🎤 COMMENTARY #{commentary_count} [Track {track_type}]")
            print(f"{'='*70}")
            print(f"{commentary}")
            print(f"{'='*70}")
            
            # Convert to voice
            if use_tts and tts_engine and audio_pipeline:
                try:
                    # Generate audio stream
                    audio_stream = tts_engine.synthesize_streaming(
                        text=commentary,
                        track_type=track_type
                    )
                    
                    # Play through pipeline
                    audio_pipeline.play_audio_stream(
                        audio_generator=audio_stream,
                        track_type=track_type,
                        text=commentary
                    )
                    
                    print(f" Audio playing...")
                    audio_pipeline.wait_until_finished(timeout=15.0)
                except Exception as e:
                    print(f" TTS Error: {e}")
            
            # Small pause between commentary
            time.sleep(0.5)
        
        # Small delay between punches
        time.sleep(0.4)
    
    # Wait for any remaining audio
    if use_tts and audio_pipeline:
        print("\n⏳ Waiting for audio to finish...")
        audio_pipeline.wait_until_finished(timeout=10.0)
    
    # Final stats
    print("\n" + "=" * 70)
    print("📊 FINAL STATISTICS")
    print("=" * 70)
    
    stats = orchestrator.get_stats()
    
    print(f"\n🥊 Fight Stats:")
    print(f"  Total Punches: {stats['total_punches']}")
    print(f"  Events Emitted: {stats['total_events_emitted']}")
    print(f"  Events Queued: {stats['total_events_queued']}")
    print(f"  Cooldown Blocks: {stats['cooldown_blocks']}")
    print(f"  Block Rate: {stats.get('block_rate', 0)*100:.1f}%")
    
    print(f"\n📝 Commentary Stats:")
    print(f"  Total Generated: {stats['commentary_generated']}")
    print(f"  Track A: {stats['track_a_generated']} ({stats['track_a_generated']/max(stats['commentary_generated'],1)*100:.1f}%)")
    print(f"  Track B: {stats['track_b_generated']} ({stats['track_b_generated']/max(stats['commentary_generated'],1)*100:.1f}%)")
    print(f"  Track A Interrupts: {stats['track_a_interrupts']}")
    
    print(f"\n🎯 Content Balance:")
    balance = stats.get('current_balance', {})
    print(f"  Track A Ratio: {balance.get('track_a', 0)*100:.1f}% (target: 60%)")
    print(f"  Track B Ratio: {balance.get('track_b', 0)*100:.1f}% (target: 40%)")
    
    print(f"\n📊 Queue Stats:")
    print(f"  Current Size: {stats.get('queue_size', 0)}")
    print(f"  Active Cooldowns: {stats.get('active_cooldowns', 0)}")
    
    print(f"\n🔧 Tracker States:")
    print(f"  Dominance: {stats['current_dominance']}")
    print(f"  Pace: {stats['current_pace']}")
    print(f"  Momentum: {stats['current_momentum']}")
    print(f"  Excitement: {stats['current_excitement']}")
    print(f"  Targets: {stats['current_targets']}")
    
    if use_tts and tts_engine:
        tts_stats = tts_engine.get_stats()
        print(f"\n🎤 TTS Stats:")
        print(f"  Characters Used: {tts_stats['total_characters']}")
        print(f"  Free Tier Usage: {tts_stats['free_tier_usage_pct']:.1f}%")
        print(f"  Avg Latency: {tts_stats['avg_first_chunk_latency_ms']:.0f}ms")
        print(f"  Success Rate: {tts_stats['success_rate']*100:.1f}%")
    
    if use_tts and audio_pipeline:
        audio_stats = audio_pipeline.get_stats()
        print(f"\n🔊 Audio Stats:")
        print(f"  Tracks Played: {audio_stats['tracks_played']}")
        print(f"  Interruptions: {audio_stats['interruptions']}")
    
    # Cleanup
    if audio_pipeline:
        audio_pipeline.shutdown()
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        simulate_fight_with_voice()
    except KeyboardInterrupt:
        print("\n\n⛔  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()