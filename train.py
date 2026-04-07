from ultralytics import YOLO
#from ultralytics import RTDETR
# model = DETR('yolov13-G2L_CRM.yaml')
model = YOLO('/root/autodl-tmp/yolov13-main/ultralytics/cfg/models/v9/yolov9s.yaml')

# Train the model
results = model.train(
  data='/root/autodl-tmp/yolov13-main/Exclusively-Dark-Image.v2i.yolov11/data.yaml',
  epochs=200, 
  batch=32, 
  imgsz=640,  
  scale=0.5,  # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
  device="0",
  patience=50,
  project="./runs",
  name="对比V9s"
)

