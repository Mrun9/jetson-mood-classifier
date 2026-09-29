import torch
from transformers import pipeline


def main():
    print("Jetson Mood Classifier\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Download once, then reuse the cached model on future runs.
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if device == "cuda" else -1,
    )

    while True:
        text = input("Enter text: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue

        result = classifier(text, truncation=True, max_length=512)[0]
        print(f"\nSentiment: {result['label']}")
        print(f"Confidence: {result['score']:.1%}\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
    print("Goodbye!")
