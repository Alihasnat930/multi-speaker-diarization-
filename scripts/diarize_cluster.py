import numpy as np
import soundfile as sf
import os
from sklearn.cluster import AgglomerativeClustering
from speechbrain.pretrained import SpeakerRecognition

# Disable symlinks on Windows to avoid privilege issues
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

MODEL_DIR = 'pretrained_models/spkrec'

def vad_segments(wav_path, sampling_rate=16000, frame_duration_ms=30):
    """
    Voice Activity Detection to find speech segments.
    Uses energy-based detection with better parameters for multi-speaker scenarios.
    """
    wav, sr = sf.read(wav_path)
    
    # Convert stereo to mono if needed
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    
    frame_len = int(sr * 0.03)  # 30ms frames
    energy = []
    
    # Calculate energy for each frame
    for i in range(0, len(wav), frame_len):
        frame = wav[i:i+frame_len]
        if len(frame) > 0:
            energy.append(np.sqrt(np.mean(frame**2)))  # RMS energy
        else:
            energy.append(0.0)
    
    energy = np.array(energy)
    
    # Dynamic threshold based on percentiles (better than mean for varying loudness)
    threshold = np.percentile(energy, 40)  # 40th percentile
    mask = energy > threshold
    
    # Apply smoothing to reduce fragmentation
    from scipy.ndimage import binary_closing
    mask = binary_closing(mask, structure=np.ones(5))  # Merge nearby segments
    
    segments = []
    start = None
    min_duration = 0.3  # Minimum segment duration in seconds
    
    for i, m in enumerate(mask):
        if m and start is None:
            start = i * frame_len
        elif not m and start is not None:
            end = i * frame_len
            duration = (end - start) / sr
            # Only keep segments longer than minimum duration
            if duration >= min_duration:
                segments.append((start/sr, end/sr))
            start = None
    
    # Handle last segment
    if start is not None:
        end = len(wav)
        duration = (end - start) / sr
        if duration >= min_duration:
            segments.append((start/sr, len(wav)/sr))
    
    print(f"VAD detected {len(segments)} voice segments")
    return segments

def diarize(wav_path, n_speakers=None):
    import torch
    from sklearn.metrics import silhouette_score
    
    spk = SpeakerRecognition.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb', 
        savedir=MODEL_DIR,
        use_auth_token=False
    )
    segs = vad_segments(wav_path)
    embeddings = []
    valid_segs = []
    
    full_audio, sr = sf.read(wav_path)
    
    for (s,e) in segs:
        # Skip segments shorter than 0.5 seconds (minimum for speaker recognition)
        if (e - s) < 0.5:
            continue
            
        start_sample = int(s*sr)
        end_sample = int(e*sr)
        audio_data = full_audio[start_sample:end_sample]
        
        # Ensure minimum length (8000 samples = 0.5 seconds at 16kHz)
        if len(audio_data) < 8000:
            # Pad with zeros to minimum length
            audio_data = np.pad(audio_data, (0, 8000 - len(audio_data)), mode='constant')
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
        emb = spk.encode_batch(audio_tensor).cpu().numpy()
        embeddings.append(emb.squeeze())
        valid_segs.append((s, e))
        
    if len(embeddings) == 0:
        return []
    
    X = np.stack(embeddings)
    
    # Normalize embeddings for better clustering
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Auto-detect number of speakers if not specified
    if n_speakers is None and len(embeddings) >= 2:
        best_score = -1
        best_n = 2
        
        # Try different numbers of speakers (2 to 5)
        for n in range(2, min(6, len(embeddings) + 1)):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n, 
                    linkage='average',
                    metric='cosine'
                ).fit(X)
                
                # Check if we actually got different clusters
                unique_labels = len(set(clustering.labels_))
                if unique_labels > 1 and unique_labels == n:
                    score = silhouette_score(X, clustering.labels_, metric='cosine')
                    print(f"Testing {n} speakers: silhouette score = {score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_n = n
            except Exception as e:
                print(f"Error testing {n} speakers: {e}")
                continue
        
        n_speakers = best_n
        print(f"Auto-detected {n_speakers} speakers (score: {best_score:.3f})")
    elif n_speakers is None:
        n_speakers = 1
    
    # Adjust n_speakers if we have fewer segments
    actual_n_speakers = min(n_speakers, len(embeddings))
    if actual_n_speakers < 2:
        # Only one segment, assign to single speaker
        print(f"Only 1 segment found, assigning to Speaker_0")
        return [{'start': valid_segs[0][0], 'end': valid_segs[0][1], 'speaker': 0}]
    
    # Use better clustering with cosine distance and normalized embeddings
    print(f"Clustering {len(embeddings)} segments into {actual_n_speakers} speakers...")
    clustering = AgglomerativeClustering(
        n_clusters=actual_n_speakers, 
        linkage='average',
        metric='cosine'
    ).fit(X)
    labels = clustering.labels_
    
    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Speaker distribution: {dict(zip(unique, counts))}")
    
    diarization = []
    for (s,e), lab in zip(valid_segs, labels):
        diarization.append({'start': s, 'end': e, 'speaker': int(lab)})
    return diarization

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python diarize_cluster.py <wav_path>')
        raise SystemExit(1)
    print(diarize(sys.argv[1]))
