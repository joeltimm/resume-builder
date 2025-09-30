#resume-builder/backend/Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir requests sentence-transformers
COPY . .
EXPOSE 5001
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]