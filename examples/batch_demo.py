"""
Simple batch processing example.
Process a recorded dental consultation and generate SOAP note.
"""
import requests
import json
import sys
from pathlib import Path


def process_audio_file(file_path: str, server_url: str = "http://localhost:8000"):
    """
    Process an audio file and get transcript + SOAP note.
    
    Args:
        file_path: Path to audio file (.wav, .mp3, .m4a)
        server_url: API server URL
    """
    # Validate file
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"\n🎤 Processing: {file_path}")
    print("=" * 70)
    
    # Upload and process
    with open(file_path, 'rb') as audio_file:
        print("📤 Uploading to server...")
        
        response = requests.post(
            f"{server_url}/process",
            files={'file': audio_file}
        )
        
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    
    # Display transcript
    print("\n📝 TRANSCRIPT")
    print("-" * 70)
    
    segments = result.get('segments', [])
    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        speaker = seg.get('speaker_id', 'Unknown')
        text = seg.get('text', '')
        confidence = seg.get('score', 0)
        
        print(f"[{start:>6.1f}s - {end:>6.1f}s] {speaker:>10s}: {text}")
        if confidence < 0.5:
            print(f"                              (⚠️  Low confidence: {confidence:.2f})")
    
    # Display SOAP note
    print("\n📋 SOAP CLINICAL NOTE")
    print("-" * 70)
    
    summary = result.get('summary', '')
    print(summary)
    
    # Save to file
    output_path = Path(file_path).stem + "_transcript.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("=" * 70 + "\n")
    
    return result


def batch_process_directory(directory: str, server_url: str = "http://localhost:8000"):
    """
    Process all audio files in a directory.
    """
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(directory).glob(f"*{ext}"))
    
    if not audio_files:
        print(f"❌ No audio files found in: {directory}")
        return
    
    print(f"\n📁 Found {len(audio_files)} audio files")
    print("=" * 70)
    
    for i, file_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {file_path.name}")
        process_audio_file(str(file_path), server_url)
    
    print("\n✅ Batch processing complete!\n")


def enroll_speaker(speaker_id: str, audio_path: str, server_url: str = "http://localhost:8000"):
    """
    Enroll a speaker via API.
    """
    print(f"\n👤 Enrolling speaker: {speaker_id}")
    
    with open(audio_path, 'rb') as audio_file:
        response = requests.post(
            f"{server_url}/enroll",
            params={'speaker_id': speaker_id},
            files={'file': audio_file}
        )
    
    if response.status_code == 200:
        print(f"✅ Successfully enrolled: {speaker_id}")
        return True
    else:
        print(f"❌ Enrollment failed: {response.status_code}")
        print(response.text)
        return False


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process dental consultation recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python batch_demo.py --file consultation_recording.mp3
  
  # Process all files in directory
  python batch_demo.py --directory ./recordings
  
  # Enroll speaker
  python batch_demo.py --enroll Dentist --audio dentist_sample.wav
  
  # Use custom server
  python batch_demo.py --file recording.wav --server http://192.168.1.100:8000
        """
    )
    
    parser.add_argument('--file', type=str, help='Process single audio file')
    parser.add_argument('--directory', type=str, help='Process all files in directory')
    parser.add_argument('--enroll', type=str, help='Enroll speaker with given ID')
    parser.add_argument('--audio', type=str, help='Audio file for enrollment')
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                       help='API server URL (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    # Check server is running
    try:
        response = requests.get(f"{args.server}/health", timeout=5)
        if response.status_code != 200:
            print(f"⚠️  Server health check failed")
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server: {args.server}")
        print("\nMake sure the server is running:")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Execute command
    if args.enroll and args.audio:
        enroll_speaker(args.enroll, args.audio, args.server)
    elif args.file:
        process_audio_file(args.file, args.server)
    elif args.directory:
        batch_process_directory(args.directory, args.server)
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("No action specified. Use --file, --directory, or --enroll")
        print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
