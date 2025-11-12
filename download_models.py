"""
Pre-download models to avoid symlink issues during runtime.
Run this script ONCE before starting the server.
"""
import os
import sys

# CRITICAL: Set this BEFORE any imports
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("=" * 60)
print("Pre-downloading SpeechBrain Models")
print("=" * 60)
print()
print("This will download models (~2GB) WITHOUT symlinks")
print("Run this ONCE, then start your server normally")
print()
print("=" * 60)
print()

try:
    from huggingface_hub import snapshot_download
    
    print("Downloading speaker recognition model...")
    print("(This may take 5-10 minutes depending on your connection)")
    print()
    
    # Download without symlinks
    cache_dir = snapshot_download(
        repo_id="speechbrain/spkrec-ecapa-voxceleb",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        local_dir="pretrained_models/spkrec",
        local_dir_use_symlinks=False  # This is the key!
    )
    
    print()
    print("✓ Models downloaded successfully!")
    print(f"Location: pretrained_models/spkrec")
    print()
    print("Now you can start the server and upload audio files.")
    print()
    
except Exception as e:
    print(f"✗ Error downloading models: {e}")
    print()
    print("Trying alternative method...")
    print()
    
    try:
        from speechbrain.pretrained import SpeakerRecognition
        
        print("Loading model (will download if needed)...")
        spk = SpeakerRecognition.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir='pretrained_models/spkrec'
        )
        
        print()
        print("✓ Model loaded successfully!")
        print()
        
    except Exception as e2:
        print(f"✗ Failed: {e2}")
        print()
        print("You may need to run this script as administrator")
        print("Or enable Developer Mode in Windows Settings")
        sys.exit(1)

print("=" * 60)
print("Setup complete! Start your server now.")
print("=" * 60)
