#!/usr/bin/env python3
"""
Script to enroll speakers (Dentist and Patient) for voice identification.
Run this before starting real-time processing to enable speaker recognition.
"""
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.spk_service import SpeakerService
from scripts.utils_audio import convert_to_wav_mono_16k
import tempfile
import joblib


def enroll_speaker(speaker_id: str, audio_path: str):
    """
    Enroll a speaker using their audio sample.
    
    Args:
        speaker_id: Identifier (e.g., 'Dentist', 'Patient', 'Dr_Smith')
        audio_path: Path to audio file with speaker's voice (10-30 seconds)
    """
    print(f"\n📝 Enrolling speaker: {speaker_id}")
    print(f"Audio file: {audio_path}")
    
    # Validate file
    if not Path(audio_path).exists():
        print(f"❌ Error: File not found: {audio_path}")
        return False
        
    # Convert to proper format
    print("Converting audio to 16kHz mono WAV...")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_wav = tmp.name
        
    try:
        convert_to_wav_mono_16k(audio_path, tmp_wav)
        
        # Initialize speaker service
        print("Loading speaker recognition model...")
        spk_service = SpeakerService()
        
        # Enroll
        print(f"Extracting speaker embedding for '{speaker_id}'...")
        spk_service.enroll(speaker_id, tmp_wav)
        
        print(f"✅ Successfully enrolled '{speaker_id}'")
        print(f"   Embedding saved to: models/enrollments/{speaker_id}.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Enrollment failed: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            Path(tmp_wav).unlink()
        except:
            pass


def list_enrolled_speakers():
    """List all enrolled speakers"""
    from pathlib import Path
    
    enroll_dir = Path('models/enrollments')
    
    if not enroll_dir.exists():
        print("\n📋 No speakers enrolled yet.")
        return
        
    enrollment_files = list(enroll_dir.glob('*.pkl'))
    
    if not enrollment_files:
        print("\n📋 No speakers enrolled yet.")
        return
        
    print(f"\n📋 Enrolled Speakers ({len(enrollment_files)}):")
    print("-" * 50)
    
    for pkl_file in enrollment_files:
        try:
            data = joblib.load(pkl_file)
            speaker_id = data.get('id', pkl_file.stem)
            embedding_shape = data.get('embedding', []).shape
            print(f"  • {speaker_id} (embedding shape: {embedding_shape})")
        except Exception as e:
            print(f"  • {pkl_file.stem} (error loading: {e})")


def interactive_enrollment():
    """Interactive enrollment mode"""
    print("\n🎤 Interactive Speaker Enrollment")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("  1. Enroll new speaker")
        print("  2. List enrolled speakers")
        print("  3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            speaker_id = input("Enter speaker ID (e.g., 'Dentist', 'Patient'): ").strip()
            audio_path = input("Enter path to audio file: ").strip()
            
            if speaker_id and audio_path:
                enroll_speaker(speaker_id, audio_path)
            else:
                print("❌ Invalid input")
                
        elif choice == '2':
            list_enrolled_speakers()
            
        elif choice == '3':
            print("\nGoodbye!")
            break
            
        else:
            print("❌ Invalid option")


def main():
    parser = argparse.ArgumentParser(
        description='Enroll speakers for voice identification in dental consultations'
    )
    parser.add_argument(
        '--id',
        type=str,
        help='Speaker ID (e.g., "Dentist", "Patient", "Dr_Smith")'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file with speaker voice (10-30 seconds recommended)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all enrolled speakers'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_enrolled_speakers()
        return
        
    # Interactive mode
    if args.interactive:
        interactive_enrollment()
        return
        
    # Command-line mode
    if args.id and args.audio:
        success = enroll_speaker(args.id, args.audio)
        sys.exit(0 if success else 1)
    else:
        # No arguments - show help
        parser.print_help()
        print("\n" + "=" * 50)
        list_enrolled_speakers()
        print("\nQuick start:")
        print("  python enroll_speaker.py --id Dentist --audio /path/to/dentist_sample.wav")
        print("  python enroll_speaker.py --id Patient --audio /path/to/patient_sample.wav")
        print("\nOr run interactively:")
        print("  python enroll_speaker.py --interactive")


if __name__ == '__main__':
    main()
