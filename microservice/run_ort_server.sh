#!/bin/bash
docker run \
    -it \
    -v "$(dirname "$PWD"):/models" \
    -p 9001:8001 \
    -p 50051:50051 \
    --name ort mcr.microsoft.com/onnxruntime/server \
    --model_path="/models/xgboost.onnx"