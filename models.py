from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor
)

def openai_whisper_small():
    # 0.2B parameters
    model_name = "openai/whisper-small"

    processor = AutoProcessor.from_pretrained(
        model_name,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.generation_config.task = "transcribe"
    model.generation_config.language = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    return processor, model

def openai_whisper_small_custom_tokenizer():
    model_name = "openai/whisper-small"

    tokenizer = WhisperTokenizerFast.from_pretrained("taiwanese_tokenizer")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    
    target_vocab_size = len(tokenizer)
    print(f"Resizing model embeddings from {model.config.vocab_size} to {target_vocab_size}")
    model.resize_token_embeddings(target_vocab_size)
    
    actual_embed_size = model.model.decoder.embed_tokens.num_embeddings
    actual_proj_out_size = model.proj_out.out_features

    model.config.use_cache = False
    model.config.vocab_size = target_vocab_size
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    
    model.generation_config.vocab_size = target_vocab_size
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.task = None
    model.generation_config.language = None
    model.generation_config.is_multilingual = False
    model.config.is_multilingual = False

    print(f"Embedding layer: {actual_embed_size}")
    print(f"Projection out: {actual_proj_out_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Generation vocab size: {model.generation_config.vocab_size}")
    
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
