FROM --platform=linux/amd64 python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      numpy==1.19.5 \
      onnx==1.8.1 \
      onnx-coreml==1.3 \
      coremltools==4.1 \
      protobuf==3.20.3 \
      onnxsim==0.4.33

ENTRYPOINT ["/bin/bash"]
