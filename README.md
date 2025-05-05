# YOLOv12-AMC2fLEFEM

YOLOv12-AMC2fLEFEM: Attention-Centric Real-Time Object Detectors With Novel Low Light Object Detection

## Install

```bash
git clone https://github.com/vmc-7645/yolov12-AMC2fLEFEM.git
cd yolov12-AMC2fLEFEM
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

## Train

```bash
yolo train model=/home/user/repos/yolov12-AMC2fLEFEM/ultralytics/cfg/models/v12/yolov12n.yaml data=/home/user/repos/LSModels/utility/dataset/data.yaml epochs=16 imgsz=640
```

## Acknowledgement

The code is based on [ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!

## Citation

```BibTeX
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

