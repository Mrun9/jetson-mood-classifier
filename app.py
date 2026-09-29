import argparse
import json
import shutil
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Conversation:
    def __init__(self, device):
        name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(name, truncation_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        self.history = []

    def reply(self, text):
        messages = [{"role": "system", "content":
                     "You are Jetson, a friendly conversational assistant. "
                     "Answer the user's questions directly; do not repeat their message. "
                     "Reply in one or two short sentences of plain English."}]
        messages += self.history + [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                truncation=True, max_length=1024).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip() or "Could you say that another way?"
        # Keep only two recent exchanges, in memory for this session.
        self.history = (messages[1:] + [{"role": "assistant", "content": answer}])[-4:]
        return answer


def predict(classifier, text, speak=False, conversation=None):
    result = classifier(text, truncation=True, max_length=512)[0]
    print(f"\nSentiment: {result['label']}")
    print(f"Confidence: {result['score']:.1%}\n")
    message = f"{result['label']}. Confidence {result['score'] * 100:.1f} percent."
    if conversation is not None:
        print("Thinking...", flush=True)
        message = conversation.reply(text)
        print(f"Assistant: {message}\n")
    if speak:
        subprocess.run(["espeak-ng", "--stdin"], input=message, text=True, check=True)


def voice_mode(classifier, microphone, conversation=None):
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
        predict(classifier, text, speak=True, conversation=conversation)


def main():
    parser = argparse.ArgumentParser(description="Local text or voice sentiment classifier.")
    parser.add_argument("--voice", action="store_true", help="listen and speak predictions")
    parser.add_argument("--chat", action="store_true", help="generate conversational replies; add --voice to speak them")
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
    conversation = Conversation(device) if args.chat else None

    if args.voice:
        voice_mode(classifier, args.mic, conversation)
        return

    while True:
        text = input("Enter text: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue

        predict(classifier, text, conversation=conversation)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print("Goodbye!")
