FROM python:3.10

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Установка зависимостей
RUN apt-get update && apt install -y \
    fonts-dejavu-core rsync git unzip tree jq moreutils aria2 wget curl \
    libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 \
    g++ build-essential pkg-config libssl-dev && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

# Установка uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh && \
    ln -s /usr/local/bin/uv /usr/bin/uv

WORKDIR /stable-diffusion-webui

# Клонирование репозитория для stable-diffusion-webui
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git  .

# Установка gdown
RUN uv pip install gdown --system

# Скачивание модели для stable-diffusion
RUN gdown https://drive.google.com/uc?id=1h3fYcL6q3KXaTC0Hdsaum8ID7VtuKYWR --fuzzy -O ./models/Stable-diffusion/realisticVisionV60B1_v51HyperVAE.safetensors

# Установка Reactor для stable-diffusion
RUN git clone https://github.com/Korpaxdev/sd-webui-reactor-sfw ./extensions/sd-webui-reactor-sfw
RUN cd ./extensions/sd-webui-reactor-sfw && uv pip install -r requirements.txt --system && \
    uv pip install onnxruntime --system

# Скачивание модели buffalo_l для insightface
RUN mkdir -p ./models/insightface/models/buffalo_l
RUN wget -q -O ./models/insightface/models/buffalo_l/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip ./models/insightface/models/buffalo_l/buffalo_l.zip -d ./models/insightface/models/buffalo_l/ && \
    rm /stable-diffusion-webui/models/insightface/models/buffalo_l/buffalo_l.zip

# Скачивание модели nsfw_detector для stable-diffusion
RUN mkdir -p ./models/nsfw_detector/vit-base-nsfw-detector
RUN wget -q -O ./models/nsfw_detector/vit-base-nsfw-detector/config.json \
    https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/config.json
RUN wget -q -O ./models/nsfw_detector/vit-base-nsfw-detector/confusion_matrix.png \
    https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/confusion_matrix.png
RUN wget -q -O ./models/nsfw_detector/vit-base-nsfw-detector/model.safetensors \
    https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/model.safetensors
RUN wget -q -O ./models/nsfw_detector/vit-base-nsfw-detector/preprocessor_config.json \
    https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/preprocessor_config.json

# Подготовка окружения для stable-diffusion
RUN python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test --xformers

# Установка зависимостей для обработчика
COPY requirements.txt .
RUN uv pip install -r requirements.txt --system

# Копирование обработчика
COPY handler.py .
# Копирование скрипта запуска
COPY start.sh .

# Установка разрешений на скрипты
RUN chmod +x start.sh

# Запуск скрипта
CMD ["./start.sh"]