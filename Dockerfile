FROM python:3.11-slim

WORKDIR /app

# System libs Pillow / torchvision need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Warm the Hugging Face cache at build time so the first /analyze request
# doesn't have to download ~700MB of Grounding DINO weights.
RUN python -c "from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; \
    AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
    AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')"

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
