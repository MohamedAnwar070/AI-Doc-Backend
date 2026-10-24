# Infrastructure Architecture

## Overview

The infrastructure can be split across two server tiers based on workload characteristics:

* **AI Inference Host** – Dedicated for heavy AI inference workloads (OCR and Voice).
* **Application Host** – Dedicated for application services, APIs, frontend, and lightweight ML services.

This separation keeps GPU/CPU-intensive inference isolated from scalable application workloads while reducing container orchestration overhead for large models.

---

# AI Inference Host

## Purpose

Hosts high-resource AI inference services:

* Runs **PaddleOCR-VL-1.5 (OCR)** microservice.
* Runs **Voice Model (STT + TTS + VAD)** microservice.
* Directly utilizes machine resources without container orchestration overhead.

## Request Flow

```text
Client / Backend
    │
    ▼
Ingress / API Gateway
    │
    ▼
Native AI Services
    ├── Paddle-VL (OCR)
    └── Voice Model
```

---

# Application Host

## Purpose

Hosts scalable application services and lightweight machine learning workloads:

* **AI Doc Retrieval**
  * Frontend (React + Vite + Nginx)
  * Backend (FastAPI)
* **Embedding Service (BGE-M3)**

## Request Flow

```text
Internet
    │
    ▼
Ingress
    │
    ▼
Application Containers
    ├── Frontend
    ├── Backend
    └── Embedding Service
```
