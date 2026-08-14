FROM python:3.12-slim

WORKDIR /app

# CPU-only torch; the default wheel drags in ~2 GB of CUDA for nothing here.
# The `||` fallback is deliberate: download.pytorch.org intermittently serves an
# empty index, failing the build with "No matching distribution found for torch"
# on a Dockerfile that built fine an hour earlier.
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
 || pip install --no-cache-dir --retries 5 --timeout 120 torch torchvision
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/eval_set.json ./data/eval_set.json
COPY reports/ ./reports/
COPY README.md ./

# Bake the weights in rather than downloading on first request: a cold Space
# otherwise spends its first minute downloading 1.5 GB while the user waits.
ENV HF_HOME=/app/.hf
RUN python -c "\
from transformers import BlipProcessor, BlipForConditionalGeneration, \
    BlipForQuestionAnswering, CLIPProcessor, CLIPModel; \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipProcessor.from_pretrained('Salesforce/blip-vqa-base'); \
BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/demo.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
