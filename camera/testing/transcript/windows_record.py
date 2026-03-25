import pyaudio
import wave

# --- Audio Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1          # 1 for mono (most USB mics are mono)
RATE = 16000          # Sample rate (16 kHz for Vosk optimization)
CHUNK = 1024          # Number of audio frames per buffer
#OUTPUT_FILE = "my_recording.wav"
OUTPUT_FILE = r"d:\Workspace\Spring\Senior Design\A-dec-Senior-Design\camera\testing\transcript\my_recording.wav"  # Name of your mono PCM .wav file


def main():
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    print("Ready to record.")
    print("Press ENTER to begin...")
    input()  # Waits for the user to press Enter

    try:
        # Open the microphone stream
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        print("🔴 Recording... (Press Ctrl+C to stop)")
        frames = []

        # Loop continuously to capture audio until interrupted
        while True:
            # exception_on_overflow=False prevents crashes if the Pi lags slightly
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

    except KeyboardInterrupt:
        # This block triggers when you press Ctrl+C
        print("\n⏹️ Recording stopped.")

    finally:
        # Clean up the audio stream gracefully
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        audio.terminate()

    # Save the captured audio data to a file
    print(f"💾 Saving audio to '{OUTPUT_FILE}'...")
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        
    print("✅ Done!")

if __name__ == "__main__":
    main()