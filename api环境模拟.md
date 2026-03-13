# 在线平台环境模拟任务

## 预期交付
1. 一个docker容器，镜像为xrd-test:latest，容器能够对宿主机提供一个API_URL，提供“读取指定目录下所有.txt文件，并生成分类预测结果，返回给API调用者”。
2. 一个宿主机中，模拟用户身份的“调用脚本”

## 预期流程——容器构建
1. 使用docker images，能够看到xrd-test:latest镜像已经拉取构建完毕
2. 创建一个容器，创建的时候就应该启用所有GPU，并提供一个端口作为API_URL（若还有其他所需的选项，可以添加）
3. 容器需要提供模型的功能，也就是模型、模型参数、所需的文件结构，需要复制并保存在容器中，具体地，[本文件所在目录./our_model/Models/]、[本文件所在目录./our_model/References/]和[本文件所在目录./our_model/run_CNN.py]需要复制到容器中，并且保持这三个文件的相对位置不变，并且在复制后的run_CNN.py的同一路径下创建一个空的Spectra目录（用于保存用户需要预测的xrd谱文件）
4. 用户（宿主机中我来模拟）通过API_URL来跟our_model或者说容器交互，我预期的交互流程是：
    > 1. 宿主机中我将需要预测的xrd谱文件放到一个文件夹中（文件夹名字可以随便取），然后通过API_URL将这个文件夹中的txt，通过文本/字符串的形式**逐个**经过api发送给容器
    > 2. 容器接收到文件夹后，将文件夹中的所有.txt文件复制到容器内的Spectra目录中（**需求更新** 舍去这一步）
    > 3. 容器内的run_CNN.py读取Spectra目录下的所有.txt文件，并生成分类预测结果 （**需求更新** 容器内部的脚本，将从api中读取到的用户待测文本，保存为Spectra/目录下的**单个txt**，然后运行模型，得到单个txt的结果并反馈给用户）
    > 4. 容器将分类预测结果返回给API调用者（至于用户拿到模型反馈后是在自己本地使用csv格式保存还是其他方式无需操心）
    > 5. 容器清空Spectra目录，为下一次预测做准备

## 用户调用api后得到的反馈格式 **需求再次更新**
### 输出规范
```json
{"code": 200|500, "status": "success|error", "message":"一般消息", "data": null}
```
说明：输出json结构。data部分自己组织，其他key基本都是固定

## 预期流程——用户调用
一个python脚本，能够模拟用户，代码中填写API_URL，然后跟容器交互，完成上述流程，得到模型输出后保存为json文件。
**需求更新：** 每次一定是通过“文本”交互，即通过api给模型/服务传输一段文本，文本内容即为需要预测的txt的完整内容，然后模型/服务会反馈这单个txt的预测结果，用户可以直接输出到屏幕也可以暂时保存，将所有待测txt都逐个给模型预测得到结果后，一起保存成json文件。
json文件的格式参考如下：
```json
{
    "BiSiTe3-3.txt": true,
    "CrSiTe3  PURE FZ40.txt": false,
    "BiSiTe3-1.txt": true,
    "CrSiTe3 pure B025-2.txt": true,
    "CrSiTe3_180_10_70_0.01_txt.txt": false,
    "Si2Te3.txt": false,
    "CrSiTe3 B027.txt": true,
    "CrSiTe3 B028.txt": true,
    "BiSiTe3-5.txt": true,
    "Cr5Te8-powder.txt": false,
    "CrAlTe-1.txt": false,
    "BiSiTe3-8.txt": true,
    "BiSiTe3-4.txt": true,
    "CrSiTe3_300_10_80_0.01.txt": false,
    "CrAlTe-2.txt": false,
    "AgCr2Te4-J113-2.txt": false,
    "CrSiTe3_1.txt": false,
    "SiTe2_164_Theoretical.txt": false,
    "SiBiTe3_148_Theoretical.txt": false,
    "BiSiTe3-6.txt": true,
    "CrSiTe3_rotation_0.01_10_80_0.01.txt": false,
    "BiSiTe3-7.txt": true,
    "CrSiTe3_148_Theoretical.txt": false,
    "CrSiTe3_2.txt": false,
    "CrTe2_JBX_J197_CT.txt": false,
    "CrSiTe3 pure B025.txt": true,
    "CrAlTe-3.txt": false,
    "CrSiTe3 pure B025-3.txt": true,
    "CrSiTe3_360_10_70_0.01_txt.txt": false,
    "CrSiTe3_back_10_70_0.01_Pr_1.txt": false,
    "BiSiTe3-2.txt": true
}
```
