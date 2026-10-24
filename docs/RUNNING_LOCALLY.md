# Running PDF OCR Backend and MongoDB Locally

This guide explains how to set up, configure, and run the MongoDB database and the PDF OCR backend server locally on Linux.

---

## 1. Local MongoDB Setup (Docker)

To run a persistent, local instance of MongoDB:

```bash
# Create a persistent Docker volume for database storage
docker volume create mongodb_data

# Start the MongoDB container on the default port (27017)
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  --restart unless-stopped \
  mongo:latest
```

### Useful Docker Commands for MongoDB
* **Check status:** `docker ps --filter name=mongodb`
* **Stop service:** `docker stop mongodb`
* **Start service (if stopped):** `docker start mongodb`
* **View logs:** `docker logs -f mongodb`

---

## 2. MongoDB Compass Setup (GUI)

To interact with your local database using a graphical interface:

### Install MongoDB Compass
Run the following commands to download and install the latest desktop client:
```bash
# Download the Compass deb package
wget https://downloads.mongodb.com/compass/mongodb-compass_1.43.0_amd64.deb

# Install the package
sudo apt install ./mongodb-compass_1.43.0_amd64.deb

# Clean up the installation file
rm mongodb-compass_1.43.0_amd64.deb
```

### Run and Connect Compass
1. Start MongoDB Compass from your terminal:
   ```bash
   mongodb-compass &
   ```
2. In the connection window, paste the local URI:
   ```text
   mongodb://localhost:27017
   ```
3. Click **Connect**.

---

## 3. Configuration & Ports

The backend ports and MongoDB connection details are configured in the [`.env`](.env) file:

```env
# Server Binding
HOST=0.0.0.0
PORT=8000          # Changed to 8000 to avoid conflicts with Paddle-VL API (8001)
RELOAD=false

# MongoDB Connection
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=pdf_ocr
```

---

## 4. Running the Backend Server

To run the FastAPI backend server, you must use the Python environment inside the virtual environment (`.venv`) where all dependencies are installed. Do not use the global `python3` command.

You can run it in one of two ways:

### Option A: Direct Execution (Recommended)
Run using the virtual environment's python directly from the project root:
```bash
.venv/bin/python main.py
```

### Option B: Activate Virtual Environment First
```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Start the server (the 'python' command now points to .venv/bin/python)
python main.py
```

The server will start up and bind to:
* **API base:** `http://localhost:8000`
* **Swagger Documentation:** `http://localhost:8000/docs`
* **ReDoc Documentation:** `http://localhost:8000/redoc`

---

## 5. Verification & Health Checks

You can verify both services are running correctly by checking the endpoints:

```bash
# Check Backend Health Status
curl -s http://localhost:8000/health
# Expected: {"status":"healthy","ocr_engine":"PaddleOCR1.5","need_voice":false}

# Check API Root
curl -s http://localhost:8000/
```
