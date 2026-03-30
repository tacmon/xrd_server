# XRD-AutoAnalyzer API 服务 (AlN 分类专版)

本项目将针对 X 射线衍射 (XRD) 数据的基于深度学习的自动分析流程，封装为了一个可供云端统一部署的 Docker API 服务。当前版本完全聚焦于识别实验样本中是否包含 **AlN (氮化铝)** 物相。

---

## 🚀 1. 快速启动服务 (服务端)

本项目使用 Docker 容器化方案，确保服务端可以在任意带有 GPU 的机器上进行一键部署。

### 1.1 构建与启动容器

在本项目**根目录**下，执行以下命令直接通过 `docker-compose` 在后台构建并启动服务：

```bash
docker compose -f docker/docker-compose.yaml up --build -d
```

> [!NOTE]
> * 该命令会自动拉取 `pytorch:2.8.0-cuda12.6-cudnn9-runtime` 基础镜像，并将核心代码环境打包。
> * 启动成功后，可以运行 `docker logs -f xrd-analyzer` 查看运行日志。
> * 服务成功启动时会提示 `Uvicorn running on http://0.0.0.0:8000`。

### 1.2 测试服务健康状态

服务运行在 `8000` 端口。你可以访问其自动生成的接口文档以确认服务健康状况：
* 在浏览器中打开：`http://localhost:8000/docs`

---

## 🛠️ 2. 用户端调用与测试 (`user_code/`)

在服务启动后，我们可以模拟前端用户行为进行批量调用测试与基准评测。

### 2.1 准备工作
确保您已经在 `user_code/.env` 中配置了目标 API 的地址（默认即为本地）：
```env
API_URL=http://localhost:8000/predict
```

### 2.2 批量调用自有 API 服务
使用我们准备好的 Python 脚本，对指定的测试集文件夹（例如 `user_code/final_data/`）进行递归验证。

```bash
cd user_code
python3 call_our_model.py --folder ./final_data --output our_model_results.json
```
> [!TIP]
> 脚本会自动遍历 `final_data` 及所有子文件夹中的 `.txt` XRD 光谱文件，并逐个发送给服务进行推理。预测过程大约耗时几分钟（每条大约 1~3 秒）。完成预测后，结果会统一保存在 `our_model_results.json` 中。

### 2.3 （可选）对比调用通用大模型
我们还提供了完全相同输入输出格式的大模型调用对照脚本。您只需提供有效的 SiliconFlow API Key 等大模型 API 配置即可：
```bash
python3 call_llm.py --folder ./final_data --key "您的_API_KEY"
```

---

## 📈 3. 模型性能评估

在执行完 API 调用并获得了预测结果 (`our_model_results.json`) 后，您可以使用评估脚本进行 F1 性能打分。
*评估脚本会自动根据文件路径中是否包含 `AlN` 来判断真实类别 (Ground Truth)。*

```bash
# 务必在 user_code 目录下执行
python3 evaluate.py --file our_model_results.json
```

系统会立刻输出如下统计报告大字报：
* 整体预测准确率 (Accuracy)
* 精确率 (Precision)
* 召回率 (Recall)
* 最终评价重点指标：**F1 分数 (F1-Score)**

---

## 📦 4. 交付给领导A（打包指南）

当您需要在内部交付给对接同事（“领导A”）以部署在腾讯云服务器上时，请注意以下几点：

1. **基础镜像替换**：请在 `docker/Dockerfile` 的第 3 行，将基础镜像改为生产环境的统一镜像版本：
   ```dockerfile
   FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
   ```
2. **源码打包发送**：目前 `docker-compose.yaml` 使用了 `build` 实时编译的形式，并未利用 Volume 映射隔离数据运行，且项目中包含模型权重文件。因此直接将该**整个项目文件夹打包（如生成 `.tar.gz` 压缩包）发送给同事**即可，无需额外生成庞大的独立 Image 镜像包传输。

```bash
# 生成交付压缩包前，请先清理一下本地容器和缓存：
docker compose -f docker/docker-compose.yaml down
rm -rf code/Novel-Space/Spectra/* code/Novel-Space/result.csv code/Novel-Space/processed_result.csv
```
---
*Author: API 服务化改造小组*
