# 安装Python 3.10和pip3
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip libgl1-mesa-glx && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 设置工作目录
WORKDIR /app

# 将当前目录的内容复制到工作目录中
COPY . /app

# 升级pip和setuptools，并安装wheel
RUN pip3 install --upgrade pip setuptools wheel

# 安装项目依赖
RUN pip3 install --no-cache-dir --ignore-installed -r requirements.txt

# 设置环境变量
ARG GLM_API_KEY
ARG YOUDAO_TTS_APP_KEY
ARG YOUDAO_TTS_APP_SECRET
ARG LIBLIB_API_AK
ARG YOUDAO_TTS_APP_SECRET

ENV GLM_API_KEY=${GLM_API_KEY}
ENV YOUDAO_TTS_APP_KEY=${YOUDAO_TTS_APP_KEY}
ENV YOUDAO_TTS_APP_SECRET=${YOUDAO_TTS_APP_SECRET}
ENV LIBLIB_API_AK=${LIBLIB_API_AK}
ENV LIBLIB_API_SK=${LIBLIB_API_SK}

# 暴露端口
EXPOSE 5001

ENTRYPOINT [ "python3.10" ]
CMD [ "app.py"]
