# train_spkrec.py (starter)
from speechbrain.pretrained import SpeakerRecognition

if __name__ == '__main__':
    spk_model = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='pretrained_models/spkrec')
    print('Loaded speaker recognition model. Example verify:')
    # Use spk_model.verify_files(file1, file2) to test speaker similarity
