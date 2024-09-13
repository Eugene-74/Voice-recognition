import whisper
import torch
import torch.nn as nn
import time

model = whisper.load_model("large")
# model = whisper.load_model("medium")

# model = whisper.load_model("base")

def translateAudio(audioFile):
    # get the time
    start = time.time()

    audio = whisper.load_audio(audioFile)
    audio = whisper.pad_or_trim(audio)

    result = model.transcribe(audioFile, language="fr", temperature=0.5, best_of=3, beam_size=5, verbose=True, fp16=False)
    # print("\n\n")
    # print(result["text"])
    # print("\n\n")


    # mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # # print(mel.shape)

    # # Créer une convolution 1x1 pour ajuster le nombre de canaux
    # conv1x1 = nn.Conv1d(in_channels=80, out_channels=128, kernel_size=1)

    # mel = conv1x1(mel)
    # # print(mel.shape)
    # # if mel.dim() == 2:  # si le tenseur est 2D (n_mels, frames)
    # #     mel = mel.unsqueeze(0)  # le rendre 3D (1, n_mels, frames)
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # options = whisper.DecodingOptions()
    # options = {
    # "language": "fr",               # Forcer la langue en français
    # "temperature": 0.5,             # Température pour ajuster la créativité de la transcription
    # "best_of": 3,                   # Générer plusieurs transcriptions et garder la meilleure
    # "beam_size": 5,                 # Utiliser un plus grand nombre de faisceaux pour la recherche (améliore la qualité)
    # "verbose": True,                # Activer les logs détaillés pour le débogage
    # "fp16": False,                  # Désactiver la précision flottante 16 bits (utile pour les machines sans GPU)
    # }

    # result = model.transcribe(audio, language="fr")
    # result = whisper.decode(model, mel, options)

    # print the recognized text
    # Print the time
    end = time.time()
    print(f"Time: {end - start}")
    return result["text"]



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


