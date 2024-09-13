import whisper

model = whisper.load_model("large")
# model = whisper.load_model("base")

def translateAudio(audioFile):

    audio = whisper.load_audio(audioFile)
    audio = whisper.pad_or_trim(audio)

    result = model.transcribe(audioFile)
    print("\n\n")
    print(result["text"])
    print("\n\n")


    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()

    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result.text



# # model = whisper.load_model("base")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("audio.opus")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()


# options = {}
# options['language'] = 'fr'
# options['verbose'] = True
# options['task'] = 'transcribe'
# # options['temperature'] = temperature
# options['best_of'] = None
# options['beam_size'] = None
# options['patience'] = None
# options['length_penalty'] = None
# options['suppress_tokens'] = "-1"
# options['initial_prompt'] = None
# options['condition_on_previous_text'] = False  # seems source of false Transcripts
# options['fp16'] = False #set false if using cpu
# options['compression_ratio_threshold'] = None #2.4
# options['logprob_threshold'] = None #-1.0 #-0.5
# options['no_speech_threshold'] = None #0.6 #0.2


