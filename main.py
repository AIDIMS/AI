import uvicorn
import torch
import numpy as np
import cv2
import timm
import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO

CONFIG = {
    "version": "AIDIMS-v2.0",
    "effnet_path": "model/best_efficientnet_b4.pth",
    "yolo_path": "model/best_yolov8m.pt",
    "threshold_effnet": 0.35,
    "threshold_yolo": 0.25,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

app = FastAPI(title="AIDIMS AI Service")


class AIEngine:
    def __init__(self):
        print(f"Khởi động AI Engine trên: {CONFIG['device']}...")
        self.device = torch.device(CONFIG["device"])

        # 1. Load EfficientNet-B4 (Sàng lọc)
        self.classifier = None
        if os.path.exists(CONFIG["effnet_path"]):
            print("Loading EfficientNet-B4...")
            self.classifier = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
            self.classifier.load_state_dict(torch.load(CONFIG["effnet_path"], map_location=self.device))
            self.classifier.to(self.device).eval()
        else:
            print(f"CẢNH BÁO: Thiếu {CONFIG['effnet_path']}. Bỏ qua bước sàng lọc.")

        # 2. Load YOLOv8 (Chẩn đoán)
        if os.path.exists(CONFIG["yolo_path"]):
            print("   -> Loading YOLOv8m...")
            self.detector = YOLO(CONFIG["yolo_path"])
        else:
            raise FileNotFoundError(f"Lỗi: Không tìm thấy model YOLO tại {CONFIG['yolo_path']}")

        print("Hệ thống sẵn sàng!")

    def preprocess_for_effnet(self, img_pil):
        img_np = np.array(img_pil)
        if img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        img_resized = cv2.resize(img_np, (380, 380))
        img_normalized = img_resized.astype(np.float32) / 255.0
        # [H, W, C] -> [1, C, H, W]
        return torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def predict(self, img_pil):
        response = {
            "model_version": CONFIG["version"],
            "classification": {"status": "UNKNOWN", "confidence": 0.0},
            "findings": []
        }

        is_abnormal = True
        eff_score = 0.0

        if self.classifier:
            img_tensor = self.preprocess_for_effnet(img_pil)
            with torch.no_grad():
                eff_score = torch.sigmoid(self.classifier(img_tensor)).item()

            response["classification"]["confidence"] = round(eff_score, 4)

            if eff_score < CONFIG["threshold_effnet"]:
                response["classification"]["status"] = "NORMAL"
                # Nếu chắc chắn bình thường, trả về luôn
                response["findings"].append({
                    "label": "No finding",
                    "confidence_score": round(1.0 - eff_score, 4),
                    "bbox_xyxy": [0, 0, 0, 0]
                })
                return response
            else:
                response["classification"]["status"] = "ABNORMAL"

        results = self.detector.predict(img_pil, conf=CONFIG["threshold_yolo"], verbose=False)[0]

        if len(results.boxes) == 0:
            response["findings"].append({
                "label": "Suspicious (Unlocalized)",
                "confidence_score": round(eff_score, 4),
                "bbox_xyxy": [0, 0, 0, 0]
            })
        else:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                label_name = results.names[cls_id]

                response["findings"].append({
                    "label": label_name,
                    "confidence_score": round(conf, 4),
                    "bbox_xyxy": [round(x, 2) for x in coords]
                })

        response["findings"].sort(key=lambda x: x['confidence_score'], reverse=True)
        return response


ai_engine = AIEngine()


@app.get("/")
def health_check():
    return {"status": "running", "version": CONFIG["version"]}


@app.post("/predict_findings")
async def api_predict(image: UploadFile = File(...)):
    # 1. Đọc file ảnh từ request
    try:
        contents = await image.read()
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File ảnh không hợp lệ: {str(e)}")

    # 2. Gọi AI xử lý
    return ai_engine.predict(img_pil)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)