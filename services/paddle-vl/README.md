# Paddle-VL Setup Guide

## 1. Set Up the Virtual Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv-paddle
source .venv-paddle/bin/activate
```

> **Note:** If you get an error that `python3-venv` is not installed, install it first:

```bash
sudo apt install python3-venv
```

---

## 2. Install the Required Dependencies

Install the project dependencies:

```bash
pip install -r requirements-paddle-vl.txt
```

Then install the PaddlePaddle CPU package:

```bash
pip install paddlepaddle
```

---

## 3. Run the API Locally

Start the Paddle-VL API server by specifying the local model directory:

```bash
PADDLE_VL_MODEL_DIR=/opt/models/paddle-vl-ocr \
GPU_ENABLED=false \
python3 paddle_vl_api.py \
    --host 0.0.0.0 \
    --port 8001
```

This will start the API server on:

```
http://0.0.0.0:8001
```