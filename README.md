# Jetson Mood Classifier

A tiny Python command-line project that labels English text **POSITIVE** or
**NEGATIVE** and prints the model's confidence. Uses Hugging Face Transformers
and the pretrained [DistilBERT sentiment model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
No training is needed. CUDA is used when PyTorch detects it; otherwise, the app uses CPU.

The model downloads on the first run (internet required). Inference runs locally.
Long inputs are truncated to 512 tokens. Confidence is a model score, not a guarantee.

## Architecture

<img src="architecture.png" alt="Text input flows through DistilBERT to a sentiment and confidence prediction. PyTorch uses CUDA when available, otherwise CPU. Inference runs locally on Jetson Orin Nano." width="800">

## Project structure

```text
jetson-mood-classifier/
├── app.py
├── architecture.png
├── requirements.txt
├── README.md
└── .gitignore
```

## Install and run

```bash
git clone https://github.com/Mrun9/jetson-mood-classifier.git
cd jetson-mood-classifier

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python3 app.py
```

**Jetson note:** GPU support may require NVIDIA's PyTorch build matched to your
JetPack and Python versions. Follow [NVIDIA's installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
to install it inside the activated virtual environment **before** installing the
requirements. A regular PyPI build may not provide Jetson CUDA support.

## Example output

```text
Jetson Mood Classifier

Device: cuda

Enter text: I really enjoyed this project!

Sentiment: POSITIVE
Confidence: 100.0%

Enter text: This was frustrating.

Sentiment: NEGATIVE
Confidence: 99.9%

Enter text: exit
Goodbye!
```

Scores may vary. Type `exit` or `quit`, or press Ctrl+C, to stop.
