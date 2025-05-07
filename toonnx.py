#!/usr/bin/env python3
# toonnx.py – safe ONNX export for AMC2fLEFEM models

from pathlib import Path
import torch, torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.amc2flefem import LEF     # adjust if path differs

# ─── config ────────────────────────────────────────────────────────────────
PT   = Path("runs/detect/train15/weights/best.pt")   # trained checkpoint
IMGSZ = (1280, 1280)                                   # image size you’ll export at
ONNX_OUT = "drone_best_dynamic.onnx"
# --------------------------------------------------------------------------

# 1️⃣  load trained model
yolo = YOLO(str(PT))
model = yolo.model

# 2️⃣  probe feature‑map shapes for every LEF
shapes = {}                   # {id(LEF) : (h, w)}
hooks  = []

def _grab_shape(module, inp, out):
    shapes[id(module)] = out.shape[2:]   # (H, W)

for m in model.modules():
    if isinstance(m, LEF):
        hooks.append(m.register_forward_hook(_grab_shape))

# dummy pass (CPU is fine)
_ = model(torch.zeros(1, 3, *IMGSZ))

for h in hooks:
    h.remove()

# 3️⃣  replace AdaptiveAvgPool2d with fixed AvgPool2d per LEF
def make_static(module: nn.Module):
    if isinstance(module, LEF):
        h, w = shapes[id(module)]
        module.pool1 = nn.AvgPool2d((h, w))
        module.pool2 = nn.AvgPool2d((max(1, h // 2), max(1, w // 2)))
        module.pool3 = nn.AvgPool2d((max(1, h // 3), max(1, w // 3)))
        module.pool4 = nn.AvgPool2d((max(1, h // 6), max(1, w // 6)))
    for c in module.children():
        make_static(c)

make_static(model)

# 4️⃣  export (dynamic H × W now allowed)
name = yolo.export(
    format   ="onnx",
    opset    =17,
    imgsz    =IMGSZ,
    batch    =1,
    dynamic  =False,
    # dynamic  =True,
    # simplify =True,
    half     =False,
    # fname    =ONNX_OUT,
)

print(f"✓ exported {name}")

print("nc =", len(YOLO(name).names))     # should be 1  (['drone'])
