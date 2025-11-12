"""
Test script to verify multi-speaker identification is working.
This will show you how the system identifies different speakers.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Multi-Speaker Identification Test")
print("=" * 60)
print()

# Check if we can import the services
try:
    from app.diarization import process_audio
    print("✓ Services loaded successfully")
except Exception as e:
    print(f"✗ Error loading services: {e}")
    sys.exit(1)

print()
print("How the system works:")
print("=" * 60)
print()
print("1. VOICE ACTIVITY DETECTION (VAD)")
print("   - Detects when someone is speaking")
print("   - Filters out silence and noise")
print()
print("2. SPEAKER DIARIZATION")
print("   - Separates audio by different speakers")
print("   - Assigns speaker labels (Speaker_0, Speaker_1, etc.)")
print()
print("3. SPEAKER IDENTIFICATION")
print("   - Uses voice embeddings (ECAPA-TDNN model)")
print("   - Compares voices to enrolled speakers")
print("   - If no enrollment: assigns generic IDs")
print()
print("4. SPEECH-TO-TEXT")
print("   - Transcribes each speaker segment")
print("   - Uses dental terminology vocabulary")
print()
print("=" * 60)
print()

# Check for audio file
import sys
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"✗ Audio file not found: {audio_file}")
        sys.exit(1)
    
    print(f"Processing: {audio_file}")
    print()
    
    try:
        results = process_audio(audio_file)
        
        print("RESULTS:")
        print("=" * 60)
        print()
        
        speakers = set()
        for i, seg in enumerate(results, 1):
            speakers.add(seg['speaker_id'])
            print(f"Segment {i}:")
            print(f"  Time: {seg['start']:.2f}s - {seg['end']:.2f}s")
            print(f"  Speaker: {seg['speaker_id']} (confidence: {seg['score']:.2f})")
            print(f"  Text: {seg['text']}")
            print()
        
        print("=" * 60)
        print(f"Total speakers detected: {len(speakers)}")
        print(f"Speakers: {', '.join(sorted(speakers))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Processing error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("To test with an audio file:")
    print(f"  python {sys.argv[0]} <path_to_audio_file.wav>")
    print()
    print("Example:")
    print(f"  python {sys.argv[0]} test_conversation.wav")
    print()
    print("Requirements:")
    print("  - Audio must be in WAV format (or install FFmpeg for MP3/M4A)")
    print("  - Should contain 2+ speakers for testing")
    print("  - Recommended: 30 seconds to 5 minutes duration")
    print()
    print("For better speaker identification:")
    print("  1. Use the /enroll endpoint to register speakers")
    print("  2. Upload 10-30 second samples of each speaker")
    print("  3. Assign names like 'Dentist', 'Patient', 'Assistant'")
