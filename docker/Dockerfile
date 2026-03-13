# 1. 拉取包含 python、cuda、pytorch 的基础镜像 (据负责人要求)
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# 2. 设置工作目录
WORKDIR /workspace

# 3. 将 docker 目录下的所有内容拷贝到容器的 /workspace/docker/ 目录下
COPY docker/ /workspace/docker/

# 4. 进入 docker 目录并安装依赖
RUN cd /workspace/docker && pip install -e .

# 5. 为了防止大模型测试和评测脚本缺少第三方包，进行一次补充安装
RUN pip install requests urllib3 pandas openpyxl fastapi uvicorn python-multipart

# 6. 将模型文件复制到容器中
COPY our_model/Models /workspace/Models
COPY our_model/References /workspace/References
COPY our_model/run_CNN.py /workspace/run_CNN.py
COPY our_model/api_server.py /workspace/api_server.py

# 7. 创建必要的目录
RUN mkdir -p /workspace/Spectra

# 8. 暴露 API 端口
EXPOSE 8000

# 9. 启动 API 服务器
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
