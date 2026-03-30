# ── XRD AutoAnalyzer API 服务镜像 ──
# 本地测试基础镜像：pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
# 提交给领导A时请改为：pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# 复制整个 code 目录内容到 /workspace
COPY code/ /workspace/

# 安装项目依赖（pip install -e . 安装 autoXRD 包）
# 同时安装 API 服务所需的额外依赖
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir fastapi uvicorn pandas

# 确保 Spectra 目录存在且为空
RUN rm -rf /workspace/Novel-Space/Spectra/* && \
    mkdir -p /workspace/Novel-Space/Spectra

# 暴露 API 服务端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["python", "/workspace/Novel-Space/server.py"]
