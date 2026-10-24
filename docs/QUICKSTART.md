# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Docker (Recommended)

```bash
# 1. Navigate to project directory
cd pdf-ocr-backend

# 2. Start the server
docker-compose up -d

# 3. Check if running
curl http://localhost:8001/health

# 4. Test with a PDF
python test_client.py your-document.pdf
```

### Option 2: Local Installation

```bash
# 1. Navigate to project directory
cd pdf-ocr-backend

# 2. Run the startup script
./run.sh

# Server will start at http://localhost:8001
```

## 📤 Upload a PDF

```bash
curl -X POST "http://localhost:8001/api/upload" \
  -F "file=@your-document.pdf"
```

**Response:**
```json
{
  "job_id": "abc-123-def",
  "filename": "your-document.pdf",
  "total_pages": 50,
  "status": "pending",
  "message": "File uploaded successfully. Processing started with paddle engine."
}
```

## 📊 Check Progress

```bash
# Replace {job_id} with your actual job ID
curl "http://localhost:8001/api/status/abc-123-def"
```

**Response:**
```json
{
  "job_id": "abc-123-def",
  "status": "running",
  "total_pages": 50,
  "processed_pages": 12,
  "percentage": 24.0,
  "engine": "paddle"
}
```

## 📥 Get Results

```bash
# Get result as JSON
curl "http://localhost:8001/api/result/abc-123-def"

# Download as text file
curl -O "http://localhost:8001/api/result/abc-123-def/download"
```

## 🔧 Switch OCR Engine

Edit `.env` file:

```env
# Use PaddleOCR (default, fast, accurate)
OCR_ENGINE=paddle

# Use Tesseract (open source)
OCR_ENGINE=tesseract

# Use Google Vision (cloud, most accurate)
OCR_ENGINE=google
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

Restart server after changing.

## ⚡ Performance Tuning

### For Fast Processing (CPU)
```env
OCR_ENGINE=paddle
USE_GPU=false
MAX_WORKERS=4  # Set to your CPU cores - 1
```

### For Maximum Accuracy
```env
OCR_ENGINE=google
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### For GPU Processing
```env
OCR_ENGINE=paddle
USE_GPU=true
MAX_WORKERS=1
```

## 🐛 Troubleshooting

### Server won't start?

```bash
# Check logs
docker-compose logs -f

# Or if running locally
cat logs/app.log
```

### Job stuck?

```bash
# Check status
curl "http://localhost:8001/api/status/{job_id}"

# Check error log
cat outputs/{job_id}/error.log
```

### Out of memory?

Reduce `MAX_WORKERS` in `.env`:
```env
MAX_WORKERS=2
```

## 📱 Using the Test Client

The included test client makes testing easy:

```bash
# Test with any PDF
python test_client.py sample.pdf

# It will:
# 1. Upload the PDF
# 2. Monitor progress
# 3. Download results
# 4. Show preview
```

## 🔗 API Documentation

Once server is running, visit:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## 💡 Common Use Cases

### Process Multiple PDFs

```bash
for pdf in *.pdf; do
  curl -X POST "http://localhost:8001/api/upload" -F "file=@$pdf"
done
```

### Monitor All Jobs

```bash
# Get all job directories
ls outputs/

# Check each status
for job_id in outputs/*/; do
  curl "http://localhost:8001/api/status/$(basename $job_id)"
done
```

### Clean Up Old Jobs

```bash
# Delete a specific job
curl -X DELETE "http://localhost:8001/api/delete/{job_id}"
```

## 🎯 Expected Performance

**320-page PDF:**
- Single process: ~3 hours
- 4 workers (CPU): ~45-60 minutes  
- GPU mode: ~30-40 minutes

**Your results may vary based on:**
- CPU/GPU specs
- PDF complexity
- OCR engine chosen
- Number of workers

## 📞 Need Help?

1. Check the full [README.md](README.md)
2. Review API docs at `/docs`
3. Open an issue on GitHub
4. Check error logs in `outputs/{job_id}/error.log`

---

**That's it! You're ready to process PDFs at scale! 🚀**
