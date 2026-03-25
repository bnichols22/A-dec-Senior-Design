import wave
import json
import sys
from vosk import Model, KaldiRecognizer

# Configuration
#MODEL_PATH = "vosk_model_heavy"     # Name of the folder containing the extracted Vosk model
MODEL_PATH = r"d:\Workspace\Spring\Senior Design\A-dec-Senior-Design\camera\testing\transcript\vosk_model_heavy"
AUDIO_FILE = r"d:\Workspace\Spring\Senior Design\A-dec-Senior-Design\camera\testing\transcript\my_recording.wav"  # Name of your mono PCM .wav file

def transcribe_audio(audio_path, model_path):
    print(f"Loading Vosk model from '{model_path}'...")
    try:
        # 1. Load the Model into memory
        model = Model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you extract the model folder to the same directory as this script?")
        sys.exit(1)

    # 2. Open the WAV file
    try:
        wf = wave.open(audio_path, "rb")
    except FileNotFoundError:
        print(f"Audio file '{audio_path}' not found.")
        sys.exit(1)

    # 3. Verify audio format (Vosk strictly requires Mono, PCM, 16-bit)
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("ERROR: Audio file must be WAV format mono PCM.")
        sys.exit(1)

    # 4. Initialize the Recognizer with the model and the audio sample rate
    rec = KaldiRecognizer(model, wf.getframerate())
    
    # Optional: Tells the recognizer to return individual word timestamps
    rec.SetWords(True) 

    print("Transcribing audio... Please wait.")
    results = []
    
    # 5. Read and process audio in small chunks (4000 frames at a time)
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        
        # AcceptWaveform returns True when a full sentence/utterance is recognized
        if rec.AcceptWaveform(data):
            # Extract the JSON result for this chunk
            part_result = json.loads(rec.Result())
            results.append(part_result.get("text", ""))

    # 6. Catch any remaining audio at the end of the file
    final_result = json.loads(rec.FinalResult())
    results.append(final_result.get("text", ""))

    # 7. Combine all the recognized sentences into one clean transcript
    # filter(None, ...) removes any empty strings from the list
    transcript = " ".join(filter(None, results))
    return transcript

if __name__ == "__main__":
    final_transcript = transcribe_audio(AUDIO_FILE, MODEL_PATH)
    print("\n" + "="*40)
    print("FINAL TRANSCRIPT:")
    print("="*40)
    print(final_transcript)