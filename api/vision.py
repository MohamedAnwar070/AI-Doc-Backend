"""Vision API — `POST /api/vision/analyze`.

Implements the "Capture photo (like Google Lens)" requirement for instant
text extraction and analysis using Google Cloud Vision API.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional, Dict, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
import httpx
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

from config import settings
from core.llm import get_ai_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])

def _get_google_vision_token() -> str:
    """Retrieve an access token for Google Cloud Vision API."""
    try:
        if settings.google_visionai_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_visionai_credentials
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(GoogleAuthRequest())
        return credentials.token
    except Exception as e:
        logger.error(f"Failed to get Google credentials: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to authenticate with Google Vision API."
        )

def _parse_vision_text(full_text_annotation: dict) -> str:
    """Parse the fullTextAnnotation from Google Vision API response into a string."""
    if not full_text_annotation or "text" in full_text_annotation:
        return full_text_annotation.get("text", "")
    
    # Manually reconstruct from blocks if 'text' isn't at root
    text_parts = []
    for page in full_text_annotation.get("pages", []):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                para_text = ""
                for word in paragraph.get("words", []):
                    word_text = "".join(symbol.get("text", "") for symbol in word.get("symbols", []))
                    para_text += word_text
                    # Check break type to add space or newline
                    last_symbol = word.get("symbols", [])[-1]
                    break_type = last_symbol.get("property", {}).get("detectedBreak", {}).get("type", "")
                    if break_type in ["SPACE", "SURE_SPACE"]:
                        para_text += " "
                    elif break_type in ["EOL_SURE_SPACE", "LINE_BREAK"]:
                        para_text += "\n"
                    elif break_type == "HYPHEN":
                        para_text += "-"
                text_parts.append(para_text.strip())
            text_parts.append("\n")
    return "\n".join(text_parts).strip()


@router.post(
    "/analyze",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file (PNG, JPEG) to analyze"),
) -> Dict[str, Any]:
    """Capture photo, extract text instantly using Google Vision API, and analyze it."""
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image file") from exc

    b64_image = base64.b64encode(content).decode("utf-8")
    token = _get_google_vision_token()

    # Call Google Vision API
    url = "https://vision.googleapis.com/v1/images:annotate"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    payload = {
        "requests": [
            {
                "features": [
                    {"maxResults": 50, "model": "builtin/latest", "type": "DOCUMENT_TEXT_DETECTION"}
                ],
                "image": {"content": b64_image}
            }
        ]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(f"Google Vision API request failed: {exc}")
            raise HTTPException(status_code=502, detail="Failed to reach Google Vision API") from exc

    vision_data = response.json()
    responses = vision_data.get("responses", [])
    if not responses:
        raise HTTPException(status_code=500, detail="Invalid response from Google Vision API")
    
    first_response = responses[0]
    
    # Error handling
    if "error" in first_response:
        err_msg = first_response["error"].get("message", "Unknown Vision API error")
        raise HTTPException(status_code=500, detail=f"Vision API Error: {err_msg}")

    full_text_annotation = first_response.get("fullTextAnnotation", {})
    extracted_text = _parse_vision_text(full_text_annotation)

    if not extracted_text:
        return {"extracted_text": "", "analysis": "No text detected in the image."}

    # Analyze via LLM to find document type, fee calculation, and legal interpretation
    llm = get_ai_client()
    system_prompt = (
        "You are an AI Domain Expert Assistant for Sub-Registrar operations. "
        "Analyze the following OCR text extracted from an uploaded image. "
        "Provide a concise summary addressing: "
        "1. Document Type "
        "2. Fee Calculation (if applicable) "
        "3. Legal Interpretation. "
        "Answer in Tamil if the text is predominantly Tamil, else answer in English."
    )
    
    try:
        analysis_result = await llm.generate_response(system_prompt, extracted_text)
    except Exception as exc:
        logger.warning(f"LLM analysis failed: {exc}")
        analysis_result = "Text extracted, but AI analysis is currently unavailable."

    return {
        "extracted_text": extracted_text,
        "analysis": analysis_result
    }
