#!/bin/bash
# Bangalore model-VM starter — same ops pattern as start_voice_api.sh / start_paddle_api.sh
#
# Install on the model VM (copy or symlink):
#   cp services/embedding/start_embedding_api.sh /opt/start_embedding_api.sh
#   chmod +x /opt/start_embedding_api.sh
#   # optional boot:
#   # crontab -e  →  @reboot /opt/start_embedding_api.sh
#
# K8s (other VM, same network) only routes traffic:
#   kubectl apply -f k8s/embedding/access-vm-port/
#   Endpoints → this host:8002 → Ingress embedding.example.com

APP_ROOT="/opt/ai-doc-retrieval-backend"
EMBED_DIR="$APP_ROOT/services/embedding"
VENV="$APP_ROOT/.venv"
LOG_FILE="$EMBED_DIR/embedding_api.log"

# Model dir on this VM (download once with download_bge_m3.py)
MODEL_PATH="${LOCAL_EMBEDDING_MODEL_PATH:-/opt/models/bge-m3}"
HOST="${EMBEDDING_HOST:-0.0.0.0}"
PORT="${EMBEDDING_PORT:-8002}"
API_KEY="${EMBEDDING_API_KEY:-changeme}"

# Boot delay only when started non-interactively (e.g. @reboot). Manual runs start immediately.
if [ ! -t 0 ] || [ "${EMBEDDING_BOOT_DELAY:-0}" = "1" ]; then
    sleep "${EMBEDDING_BOOT_SLEEP:-20}"
fi

# Avoid starting duplicate process
if pgrep -f "$EMBED_DIR/embedding_api.py" > /dev/null; then
    echo "$(date): embedding_api.py already running" >> "$LOG_FILE"
    exit 0
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "$(date): ERROR model path missing: $MODEL_PATH" >> "$LOG_FILE"
    echo "$(date): Run: cd $EMBED_DIR && source $VENV/bin/activate && python3 download_bge_m3.py --target_dir $MODEL_PATH" >> "$LOG_FILE"
    exit 1
fi

cd "$APP_ROOT" || exit 1
# Prefer dedicated embedding venv (avoids main .venv hub/ST version fights)
if [ -f "$APP_ROOT/.venv-embedding/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$APP_ROOT/.venv-embedding/bin/activate"
elif [ -f "$VENV/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
else
    echo "$(date): ERROR no venv at $VENV or .venv-embedding" >> "$LOG_FILE"
    exit 1
fi

cd "$EMBED_DIR" || exit 1

export LOCAL_EMBEDDING_MODEL_PATH="$MODEL_PATH"
export EMBEDDING_API_KEY="$API_KEY"
export PORT="$PORT"

nohup python3 embedding_api.py --host "$HOST" --port "$PORT" --model_path "$MODEL_PATH" \
    >> "$LOG_FILE" 2>&1 &

echo "$(date): embedding_api.py started host=$HOST port=$PORT model=$MODEL_PATH" >> "$LOG_FILE"
