"""
Standalone Local Embedding API Server.
Exposes an OpenAI-compatible /v1/embeddings endpoint using SentenceTransformers.

Running this service:

python embedding_api.py --host 0.0.0.0 --port 8002 --model_path "/opt/models/bge-m3"

"""

import os
import argparse
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Union, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embedding_api")

# DEFAULT_MODEL_PATH = "/Users/gsmmgsm/gsm/ai-ml/ai-orginal/ai-models/bge-m3"
# DEFAULT_MODEL_PATH = "D:\models\bge-m3"
DEFAULT_MODEL_PATH = "/opt/models/bge-m3"

# We use globals to store the loaded model and its resolved path.
model: Optional[SentenceTransformer] = None
loaded_model_path: Optional[str] = None

# Security
API_KEY = os.environ.get("EMBEDDING_API_KEY")
security = HTTPBearer(auto_error=False)


def get_model_path() -> str:
    return os.environ.get(
        "LOCAL_EMBEDDING_MODEL_PATH",
        DEFAULT_MODEL_PATH,
    )


def get_model_name() -> str:
    if loaded_model_path:
        return os.path.basename(os.path.normpath(loaded_model_path))

    if model is None:
        return "unknown"

    for attr in ("model_name", "model_name_or_path"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return os.path.basename(os.path.normpath(value))

    return "unknown"


def load_model(model_path: str) -> SentenceTransformer:
    global loaded_model_path

    logger.info("Loading model from: %s", model_path)
    loaded_model = SentenceTransformer(model_path)
    loaded_model_path = model_path

    if not hasattr(loaded_model, "model_name"):
        loaded_model.model_name = os.path.basename(os.path.normpath(model_path))

    logger.info("Warming up model...")
    loaded_model.encode(["warm up"])
    logger.info("Model loaded and warmed up successfully!")
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if model is None:
        try:
            model = load_model(get_model_path())
        except Exception as e:
            logger.error("Failed to load model during startup: %s", e)
            raise RuntimeError(
                f"Failed to load model from {get_model_path()}: {e}"
            ) from e

    yield


# ── CONFIG ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Local Embedding API", version="1.0.0", lifespan=lifespan)


def verify_api_key(request: Request, auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Verifies the API key. Supports:
    1. Bearer Token (for Swagger/Postman)
    2. Basic Auth (for backward compatibility with pdf-ocr-backend)
    """
    if not API_KEY:
        return None
        
    header = request.headers.get("Authorization")
    if not header:
        raise HTTPException(status_code=403, detail="Invalid authentication credentials")

    # 1. Check Bearer (used by Swagger UI "Authorize" button)
    if header.startswith("Bearer "):
        token = header.split(" ")[1]
        if token == API_KEY:
            return token

    # 2. Check Basic Auth (used by your pdf-ocr-backend: user:your_api_key)
    elif header.startswith("Basic "):
        try:
            import base64
            decoded = base64.b64decode(header.split(" ")[1]).decode()
            if API_KEY in decoded.split(":"):
                return API_KEY
        except Exception:
            pass

    logger.warning("Unauthorized access attempt.")
    raise HTTPException(status_code=403, detail="Invalid authentication credentials")


# ── MODELS ───────────────────────────────────────────────────────────────────
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "bge-m3"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


# ── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "ok" if model is not None else "degraded",
        "model": get_model_name(),
        "model_path": loaded_model_path or get_model_path(),
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
@app.post("/embeddings", response_model=EmbeddingResponse)
def create_embeddings(
    request: EmbeddingRequest, api_key: str = Depends(verify_api_key)
):
    """
    Generate embeddings for the provided input text(s).
    Matches OpenAI's request/response format.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Standardize input to a list of strings
    inputs = [request.input] if isinstance(request.input, str) else request.input

    try:
        # Generate embeddings
        # convert_to_numpy=True + tolist() is generally efficient
        embeddings = model.encode(inputs).tolist()

        # Build response data
        data = [
            EmbeddingData(index=i, embedding=emb) for i, emb in enumerate(embeddings)
        ]

        # Simple token count estimation (4 chars per token)
        total_chars = sum(len(text) for text in inputs)
        estimated_tokens = total_chars // 4

        response = EmbeddingResponse(
            data=data,
            model=request.model or "bge-m3",
            usage={
                "prompt_tokens": estimated_tokens,
                "total_tokens": estimated_tokens,
                "ms_taken": int((time.time() - start_time) * 1000),
            },
        )

        logger.info(
            "Generated %d embeddings for %d tokens in %.2fms",
            len(data),
            estimated_tokens,
            (time.time() - start_time) * 1000,
        )
        return response

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── STARTUP ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Start local embedding API")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8002, help="Listen port")
    parser.add_argument(
        "--model_path",
        default=get_model_path(),
        help="Path to the sentence-transformer model folder",
    )
    args = parser.parse_args()

    os.environ["LOCAL_EMBEDDING_MODEL_PATH"] = args.model_path

    uvicorn.run("embedding_api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
