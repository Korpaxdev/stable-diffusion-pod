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

# Установка зависимостей для stable-diffusion-webui
RUN uv pip install -r requirements.txt --system

# Установка gdown
RUN uv pip install gdown --system

# Скачивание модели для stable-diffusion
RUN gdown https://drive.google.com/uc?id=1h3fYcL6q3KXaTC0Hdsaum8ID7VtuKYWR --fuzzy -O ./models/Stable-diffusion/realisticVisionV60B1_v51HyperVAE.safetensors

# Установка Reactor для stable-diffusion
RUN git clone https://github.com/Korpaxdev/sd-webui-reactor-sfw ./extensions/sd-webui-reactor-sfw
RUN cd ./extensions/sd-webui-reactor-sfw && uv pip install -r requirements.txt --system && \
    uv pip install onnxruntime --system

RUN chmod +x start.sh

CMD ["./start.sh"]