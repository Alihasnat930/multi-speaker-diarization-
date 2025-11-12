import os
import joblib
import numpy as np
from pathlib import Path

# Disable symlinks on Windows to avoid privilege issues - MUST be before imports
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Monkey-patch Path.symlink_to to use copy instead (Windows workaround)
import shutil
from pathlib import Path as OrigPath

_original_symlink_to = OrigPath.symlink_to

def _copy_instead_of_symlink(self, target, target_is_directory=False):
    """Replace symlink with copy on Windows to avoid privilege errors"""
    try:
        # Try original symlink first
        return _original_symlink_to(self, target, target_is_directory)
    except OSError as e:
        if "WinError 1314" in str(e) or "privilege" in str(e).lower():
            # Copy file instead of symlink
            if isinstance(target, (str, OrigPath)):
                target = OrigPath(target)
            if target.exists():
                if target.is_file():
                    self.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, self)
                    print(f"Copied (instead of symlink): {target.name}")
                elif target.is_dir():
                    shutil.copytree(target, self, dirs_exist_ok=True)
            return self
        else:
            raise

OrigPath.symlink_to = _copy_instead_of_symlink

# Import after setting environment variables and monkey-patch
from speechbrain.pretrained import SpeakerRecognition
from huggingface_hub import snapshot_download

ENROLL_DIR = Path('models/enrollments')
MODEL_DIR = 'pretrained_models/spkrec'

class SpeakerService:
    def __init__(self):
        # Ensure model directory exists
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        
        # Load model - handle symlink errors by copying files
        try:
            # Try loading from local directory first if files exist
            if (Path(MODEL_DIR) / "hyperparams.yaml").exists():
                print(f"Loading speaker model from {MODEL_DIR}...")
                self.spk = SpeakerRecognition.from_hparams(
                    source=MODEL_DIR,
                    savedir=MODEL_DIR,
                    use_auth_token=False
                )
            else:
                # Download from HuggingFace
                self.spk = SpeakerRecognition.from_hparams(
                    source='speechbrain/spkrec-ecapa-voxceleb', 
                    savedir=MODEL_DIR,
                    use_auth_token=False
                )
        except OSError as e:
            if "WinError 1314" in str(e) or "privilege" in str(e).lower():
                # Symlink error - manually copy files from cache
                print("Symlink error detected - copying files instead...")
                import shutil
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--speechbrain--spkrec-ecapa-voxceleb"
                
                if cache_dir.exists():
                    # Find the snapshot directory
                    snapshots = list((cache_dir / "snapshots").glob("*"))
                    if snapshots:
                        src = snapshots[0]
                        dst = Path(MODEL_DIR)
                        dst.mkdir(parents=True, exist_ok=True)
                        
                        # Copy all files
                        for item in src.rglob("*"):
                            if item.is_file():
                                rel_path = item.relative_to(src)
                                dest_file = dst / rel_path
                                dest_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, dest_file)
                        
                        print(f"✓ Copied model files to {MODEL_DIR}")
                        
                        # Now load from local directory - may still have symlink issues
                        try:
                            self.spk = SpeakerRecognition.from_hparams(
                                source=MODEL_DIR,
                                savedir=MODEL_DIR,
                                use_auth_token=False
                            )
                        except OSError as e2:
                            if "WinError 1314" in str(e2):
                                # Still having symlink issues - try one more copy
                                print("Copying remaining files...")
                                for item in src.rglob("*"):
                                    if item.is_file():
                                        rel_path = item.relative_to(src)
                                        dest_file = dst / rel_path
                                        if not dest_file.exists():
                                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                                            shutil.copy2(item, dest_file)
                                
                                # Final attempt
                                self.spk = SpeakerRecognition.from_hparams(
                                    source=MODEL_DIR,
                                    savedir=MODEL_DIR,
                                    use_auth_token=False
                                )
                            else:
                                raise
                    else:
                        raise Exception("Model cache not found - please download manually")
                else:
                    raise
            else:
                raise
        ENROLL_DIR.mkdir(parents=True, exist_ok=True)
        self.enrollments = {}
        self.load_enrollments()

    def load_enrollments(self):
        enrollments = {}
        for p in ENROLL_DIR.glob('*.pkl'):
            d = joblib.load(p)
            enrollments[d['id']] = d['embedding']
        self.enrollments = enrollments

    def match_embedding(self, emb):
        best = None
        best_score = -1
        for id_, v in self.enrollments.items():
            sim = np.dot(emb, v) / (np.linalg.norm(emb) * np.linalg.norm(v))
            if sim > best_score:
                best_score = sim; best = id_
        return best, best_score

    def get_embedding_file(self, wav_path):
        # Use encode_batch method for SpeakerRecognition
        # Use soundfile instead of torchaudio to avoid torchcodec dependency
        import soundfile as sf
        import torch
        from scipy import signal as scipy_signal
        
        # Load audio with soundfile
        audio_data, fs = sf.read(wav_path)
        
        # Resample if needed
        if fs != 16000:
            # Calculate number of samples in resampled audio
            num_samples = int(len(audio_data) * 16000 / fs)
            audio_data = scipy_signal.resample(audio_data, num_samples)
        
        # Convert to torch tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
        
        # Get embedding
        emb = self.spk.encode_batch(audio_tensor)
        return emb.squeeze().cpu().numpy()

    def enroll(self, speaker_id, wav_path):
        emb = self.get_embedding_file(wav_path)
        joblib.dump({'id': speaker_id, 'embedding': emb}, ENROLL_DIR / f'{speaker_id}.pkl')
        self.load_enrollments()
