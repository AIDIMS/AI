import uvicorn
import torch
import numpy as np
import cv2
import timm
import io
import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO

CONFIG = {
    "version": "AIDIMS-v2.0",
    "effnet_path": "model/best_efficientnet_b4.pth",
    "yolo_path": "model/best_yolov8m_fold4.pt",
    "threshold_effnet": 0.35,
    "threshold_yolo": 0.16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "debug_folder": "debug_images"
}

os.makedirs(CONFIG["debug_folder"], exist_ok=True)

app = FastAPI(title="AIDIMS AI Service")


class AIEngine:
    def __init__(self):
        print(f"Khởi động AI Engine trên: {CONFIG['device']}...")
        self.device = torch.device(CONFIG["device"])

        self.classifier = None
        if os.path.exists(CONFIG["effnet_path"]):
            print("Loading EfficientNet-B4...")
            self.classifier = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
            self.classifier.load_state_dict(torch.load(CONFIG["effnet_path"], map_location=self.device))
            self.classifier.to(self.device).eval()
        else:
            print(f"CẢNH BÁO: Thiếu {CONFIG['effnet_path']}. Bỏ qua bước sàng lọc.")

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
        return torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def save_debug_image(self, img_array, prefix="pred"):

        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.jpg"
        save_path = os.path.join(CONFIG["debug_folder"], filename)

        if img_array.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        cv2.imwrite(save_path, img_bgr)
        print(f"Đã lưu ảnh debug tại: {save_path}")
        return filename

    def predict(self, img_pil):
        response = {
            "model_version": CONFIG["version"],
            "classification": {"status": "UNKNOWN", "confidence": 0.0},
            "findings": [],
            "debug_image": None
        }

        eff_score = 0.0

        if self.classifier:
            img_tensor = self.preprocess_for_effnet(img_pil)
            with torch.no_grad():
                eff_score = torch.sigmoid(self.classifier(img_tensor)).item()

            response["classification"]["confidence"] = round(eff_score, 4)

            if eff_score < CONFIG["threshold_effnet"]:
                response["classification"]["status"] = "NORMAL"
                response["findings"].append({
                    "label": "No finding",
                    "confidence_score": round(1.0 - eff_score, 4),
                    "bbox_xyxy": [0, 0, 0, 0]
                })

                img_vis = np.array(img_pil)
                cv2.putText(img_vis, f"NORMAL ({round(1.0 - eff_score, 2)})", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

                saved_name = self.save_debug_image(img_vis, prefix="NORMAL")
                response["debug_image"] = saved_name

                return response
            else:
                response["classification"]["status"] = "ABNORMAL"

        results = self.detector.predict(
            img_pil,
            conf=CONFIG["threshold_yolo"],
            imgsz=1024,
            verbose=False
        )[0]

        annotated_frame = results.plot(labels=True, boxes=True, font_size=20, line_width=3)


        timestamp = int(time.time() * 1000)
        filename = f"ABNORMAL_{timestamp}.jpg"
        save_path = os.path.join(CONFIG["debug_folder"], filename)
        cv2.imwrite(save_path, annotated_frame)

        response["debug_image"] = filename
        print(f"Đã lưu ảnh YOLO Visualize tại: {save_path}")

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
                coords = box.xyxy[0].tolist()
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
    try:
        contents = await image.read()
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File ảnh không hợp lệ: {str(e)}")

    return ai_engine.predict(img_pil)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)