import whisper
# import torch
import torch.nn as nn
import time



def translateAudio(audioFile):

    audio = whisper.load_audio(audioFile)
    # audio = whisper.pad_or_trim(audio)

    result = model.transcribe(audioFile, language="es", temperature=0.5, best_of=1, beam_size=5, verbose=True, fp16=False)


    return result["text"]

if __name__ == "__main__":
    model = whisper.load_model("large")
    # model = whisper.load_model("medium")
    # model = whisper.load_model("base")

    # audio_file = "audio.mp3"
    # transcription = translateAudio(audio_file)
    # for i in range(7,7+1):
    # transcription = translateAudio(str(i)+".mp3")
    transcription = translateAudio("WhatsApp Ptt 2024-11-05 at 22.09.10.mp3")

    
    print(f"Transcription: {transcription}")

