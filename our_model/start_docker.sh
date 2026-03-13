#!/bin/bash

# 获取当前脚本所在目录的绝对路径 (即 our_model 目录)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "======================================================"
echo "正在使用 xrd-test:latest 镜像运行环境"
echo "即将执行指令: python run_CNN.py --inc_pdf"
echo "（运行期间将在此处实时打印输出，运行结束后容器会自动销毁）"
echo "======================================================"

# 运行 docker 容器 (前台执行模式)
# 参数说明:
#   -i      : 保持标准输入打开，以便在需要时进行交互
#   -t      : 分配一个伪终端，使输出有颜色和正确的格式
#   --rm    : 容器退出（包括内部进程如python结束）后自动删除，保持环境整洁
#   -v      : 将本地 our_model 目录挂载到容器内的 /workspace
#   -w      : 指定容器启动后的默认工作目录为 /workspace
docker run -it --rm \
    -v "$DIR":/workspace \
    -w /workspace \
    xrd-test:latest \
    python run_CNN.py --inc_pdf

echo "======================================================"
echo "✅ 任务执行完毕！容器已自动清理。"
echo "======================================================"
