# Architecture Notes

- ASR: Use SpeechBrain ASR recipes for fine-tuning on dental speech; export best checkpoint to pretrained_models/asr.
- Speaker recognition: ECAPA-TDNN with enrollment data stored in models/enrollments.
- Diarization: Current implementation uses simple VAD + embedding clustering. Replace with pyannote or a neural diarizer for production.
- Summarization: Fine-tune an instruction model (LLM) for transcript->SOAP conversion. Use clinician review.
