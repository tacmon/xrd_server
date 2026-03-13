#!/bin/bash

# 获取当前脚本所在目录的绝对路径 (即 our_model 目录)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "======================================================"
echo "🚀 正在启动 XRD API 模拟服务..."
echo "API 地址: http://localhost:8000/predict"
echo "按 Ctrl+C 可以停止并销毁容器"
echo "======================================================"

# 运行 docker 容器
# -p 8000:8000 : 端口映射
# --name xrd-api-server : 给容器取个名字方便查找
docker run -it --rm \
    --name xrd-api-server \
    -p 8000:8000 \
    -v "$DIR":/workspace \
    -w /workspace \
    xrd-test:latest \
    uvicorn api_server:app --host 0.0.0.0 --port 8000
