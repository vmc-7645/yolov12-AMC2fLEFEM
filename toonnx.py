from ultralytics import YOLO

model = YOLO("runs/train/yolov12n.pt")     # your trained weights

# ‑‑ export with everything you usually want for production
model.export(
    format   = "onnx",
    opset    = 17,          # safe for ORT ≥1.14; 18 is also fine
    device   = "cuda",      # export on GPU → faster when model is big
    imgsz    = (864, 1504), # or use a square if you trained square
    batch    = 1,           # 1‑batch inference is what you benchmark
    dynamic  = True,        # allow any resolution
    simplify = True,        # calls onnxsim under the hood
    half     = False,       # leave in FP32; you can quantize later
)
# => writes drone_best.onnx

onnx_model = YOLO("yolov12n.onnx")
print("classes in ONNX:", len(onnx_model.names))
