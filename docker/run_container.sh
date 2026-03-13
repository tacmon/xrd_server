#!/bin/bash
# 运行环境模拟容器的脚本

IMAGE_NAME="xrd-test:latest"
CONTAINER_NAME="xrd-api-server"
PORT=8000

echo "🚀 正在启动容器 ${CONTAINER_NAME}..."

# 检查容器是否已经在运行，如果是则停止并删除
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "🔄 发现同名容器正在运行/存在，正在停止并移除..."
    docker stop ${CONTAINER_NAME} > /dev/null
    docker rm ${CONTAINER_NAME} > /dev/null
fi

# 启动容器
# --gpus all: 启用所有 GPU
# -p 8000:8000: 映射 API 端口
# -d: 后台运行
docker run --gpus all \
    -p ${PORT}:${PORT} \
    --name ${CONTAINER_NAME} \
    -d \
    ${IMAGE_NAME}

if [ $? -eq 0 ]; then
    echo "================================================="
    echo "✅ 容器已成功启动！"
    echo "API 地址: http://localhost:${PORT}/predict"
    echo "您现在可以使用 client.py 进行调用了。"
    echo "================================================="
else
    echo "❌ 容器启动失败，请检查 Docker 环境或 GPU 驱动。"
    exit 1
fi
