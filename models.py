from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def openai_whisper_small():
    # 0.2B parameters
    model_name = "openai/whisper-small"

    processor = AutoProcessor.from_pretrained(
        model_name,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        task="transcribe"
    )
    return processor, model

def openai_whisper_medium():
    # 0.8B parameters
    model_name = "openai/whisper-medium"

    processor = AutoProcessor.from_pretrained(
        model_name,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        task="transcribe"
    )
    return processor, model
