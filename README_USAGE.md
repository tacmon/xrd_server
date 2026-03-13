# XRD 封装组件使用说明

本指南旨在指导您如何启动 API 模拟服务并在宿主机运行测试。

## 1. 启动 API 模拟服务（容器）

我们为您准备了一个便捷的启动脚本，它会启动 Docker 容器并自动暴露接口。

**执行步骤：**
```bash
cd /home/tacmon/材料大模型-xrd-封装组件/our_model
bash start_api.sh
```

> [!TIP]
> 启动成功后，您会看到 `Uvicorn running on http://0.0.0.0:8000` 的提示。请保持该终端窗口打开，不要关闭。

---

## 2. 在宿主机中进行模拟测试

当 API 服务启动后，您可以在**另一个终端窗口**中运行模拟客户端进行测试。

**执行步骤：**
```bash
cd /home/tacmon/材料大模型-xrd-封装组件/our_model
python3 client.py --folder ./Spectra --api http://localhost:8000/predict
```

### 参数说明：
- `--folder`: 存放待测 `.txt` 文件的目录（默认是 `./Spectra`）。
- `--api`: 服务地址（默认是 `http://localhost:8000/predict`）。
- `--output`: 最终结果保存的文件名（默认是 `results.json`）。

---

## 3. 结果查看

测试完成后，客户端会生成一个 `results.json` 文件。
该文件会汇聚所有测试文件的预测结果，格式符合最新的需求规范：

```json
{
    "xxx.txt": true,
    "yyy.txt": false,
    ...
}
```

## 4. 常见问题排查

- **Connection refused**: 请确保您已经运行了 `start_api.sh` 并且容器正在运行。
- **镜像未找到**: 如果提示找不到 `xrd-test:latest`，请先运行 `cd ../docker && bash build` 重新构建镜像。
