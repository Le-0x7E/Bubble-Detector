## 项目简介
本项目的主要目标是进行液体中气泡的检测，项目所应用的检测模型是使用包含约一万张水体气泡图像的数据集对YOLOv8n进行训练得到的。

本项目的GUI可选择图片或视频文件作为输入，并集成了多项检测结果的显示，包括气泡二维面积占比、气泡数量等，下图为示例：
![demo](https://le.0x7e.tech/wp-content/uploads/2025/02/GUI.png)
## 如何运行

 - 请先在Python>=3.8的环境中安装项目相关依赖包：
```bash
pip install -r requirements.txt
```
 - 然后运行app.py即可：
```bash
python app.py
```
 - 项目文件夹./demo_img中包含了两张检测示例图片，可供测试使用。

## 注意事项
- ultralytics遵循AGPL-3.0，如需商业用途，需要取得其license。
- 如需使用自己训练的YOLOv8模型，需要将.pt模型文件放入./models文件夹中。
- 可保存检测结果，保存路径为./run。