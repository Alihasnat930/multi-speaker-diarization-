"""
Quick test to verify the HF_HUB_DISABLE_SYMLINKS fix is working
"""
import os
import sys

print("=" * 60)
print("Testing HuggingFace Symlink Configuration")
print("=" * 60)
print()

# Check environment variable
hf_disable = os.environ.get('HF_HUB_DISABLE_SYMLINKS', 'NOT SET')
print(f"HF_HUB_DISABLE_SYMLINKS = {hf_disable}")
print()

if hf_disable == '1':
    print("✓ Symlinks are DISABLED (correct for Windows)")
    print("  Models will be copied instead of symlinked")
else:
    print("✗ Symlinks are NOT disabled")
    print("  This may cause [WinError 1314] on Windows")
    print()
    print("Fix: Set environment variable before importing SpeechBrain:")
    print("  import os")
    print("  os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'")

print()
print("=" * 60)
print()

# Try to import and test
print("Testing SpeechBrain import...")
try:
    # Set the variable in case it's not set
    os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
    
    from speechbrain.pretrained import SpeakerRecognition
    print("✓ SpeechBrain imported successfully")
    print()
    
    print("Note: When models are downloaded, they will:")
    if os.environ.get('HF_HUB_DISABLE_SYMLINKS') == '1':
        print("  ✓ Be COPIED (no admin rights needed)")
    else:
        print("  ✗ Use SYMLINKS (needs admin rights)")
    
except Exception as e:
    print(f"✗ Error: {e}")

print()
print("=" * 60)
