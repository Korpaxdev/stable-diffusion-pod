#!/bin/bash

echo "Starting Automatic1111 WebUI..."

# Переходим в директорию WebUI
cd /stable-diffusion-webui

# Запускаем WebUI в фоне с API
python launch.py \
  --listen \
  --port 7860 \
  --api \
  --skip-torch-cuda-test \
  --xformers \
  --disable-model-loading-ram-optimization \
  --enable-insecure-extension-access \
  --no-half-vae \
  --opt-split-attention \
  --disable-console-progressbars \
  --skip-version-check \
  --no-half-vae --precision full --no-half \
  --no-download-sd-model &

# Ждем немного для запуска WebUI
sleep 10

echo "Starting RunPod handler..."

# # Переходим обратно в корневую директорию и запускаем обработчик
# cd /
# python handler.py