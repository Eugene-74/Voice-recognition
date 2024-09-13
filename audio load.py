import pyaudio
import wave
import translate

from pydub import AudioSegment


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def start_recording(output_filename="output.wav"):
    audio = pyaudio.PyAudio()
    
    # Ouvrir un flux pour le microphone
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("Enregistrement démarré... Appuyez sur 'stop' pour arrêter.")
    frames = []
    
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print("Enregistrement arrêté.")

    # Arrêter et fermer le flux
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Sauvegarder en WAV
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Fichier sauvegardé sous : {output_filename}")

def show_menu():
    while True:
        print("\nMenu :")
        print("1. Démarrer l'enregistrement")
        print("2. Démarrer la transcription")
        print("3. Quitter")
        
        choice = input("Choisissez une option : ")
        filename = "audio.mp3"
        if choice == '1':
            # filename = input("Entrez le nom du fichier de sortie (par défaut 'output.wav'): ")
            # if not filename:
            
            start_recording(output_filename=filename)
        elif choice == '2':
            translate.translateAudio(filename)
        elif choice == '3':
            print("Quitter le programme.")
            break
        else:
            print("Option invalide. Veuillez réessayer.")

if __name__ == "__main__":
    show_menu()

# Utiliser la fonction pour enregistrer pendant 5 secondes



