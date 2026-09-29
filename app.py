import argparse
import json
import shutil
import subprocess
import sys

import torch
from transformers import pipeline


def predict(classifier, text, speak=False):
    result = classifier(text, truncation=True, max_length=512)[0]
    print(f"\nSentiment: {result['label']}")
    print(f"Confidence: {result['score']:.1%}\n")
    if speak:
        message = f"{result['label']}. Confidence {result['score'] * 100:.1f} percent."
        subprocess.run(["espeak-ng", "--stdin"], input=message, text=True, check=True)


def voice_mode(classifier, microphone):
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model

    # Vosk runs on CPU and downloads this small English model only once.
    model = Model(model_name="vosk-model-small-en-us-0.15")
    rate = int(sd.query_devices(microphone, "input")["default_samplerate"])
    recognizer = KaldiRecognizer(model, rate)

    while True:
        recognizer.Reset()
        print("Listening... Speak English, then pause. Say 'exit' or 'quit' to stop.")
        # Close the microphone before prediction and speech to avoid feedback.
        with sd.RawInputStream(samplerate=rate, device=microphone,
                               channels=1, dtype="int16") as stream:
            previous = ""
            while True:
                audio, overflowed = stream.read(max(1, rate // 5))
                if overflowed:
                    raise RuntimeError("Microphone audio overflow. Close other apps and retry.")
                if recognizer.AcceptWaveform(bytes(audio)):
                    text = json.loads(recognizer.Result()).get("text", "").strip()
                    if text:
                        break
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial and partial != previous:
                        print(f"Hearing: {partial}", flush=True)
                        previous = partial

        print(f"\nYou said: {text}")
        if text.lower() in ("exit", "quit"):
            break
        predict(classifier, text, speak=True)


def main():
    parser = argparse.ArgumentParser(description="Local text or voice sentiment classifier.")
    parser.add_argument("--voice", action="store_true", help="listen and speak predictions")
    parser.add_argument("--mic", type=int, help="input device number from --list-devices")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.voice and not shutil.which("espeak-ng"):
        raise RuntimeError("Voice output needs espeak-ng: sudo apt install espeak-ng")

    print("Jetson Mood Classifier v2\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Download once, then reuse the cached model on future runs.
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if device == "cuda" else -1,
    )

    if args.voice:
        voice_mode(classifier, args.mic)
        return

    while True:
        text = input("Enter text: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue

        predict(classifier, text)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print("Goodbye!")
