import torch
import numpy as np
import io
import os
import cv2
import timm
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO
import uvicorn

MODEL_VERSION = "AIDIMS-v2.0"
EFFNET_PATH = "model/best_efficientnet_b4.pth"
YOLO_PATH = "model/best_yolov8m.pt"

CLASSES_LIST = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
    'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity', 'Lung cyst',
    'Mediastinal shift', 'Nodule/Mass', 'Other lesion', 'Pleural effusion',
    'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture'
]

THRESHOLD_EFFNET = 0.35
THRESHOLD_YOLO = 0.25

app = FastAPI(title="AIDIMS AI Service")


# --- HÀM LOAD MODEL ---
class AI_System:
    def __init__(self):
        print("⚙Đang khởi tạo hệ thống AIDIMS...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        print("Loading EfficientNet-B4...")
        self.classifier = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
        if os.path.exists(EFFNET_PATH):
            self.classifier.load_state_dict(torch.load(EFFNET_PATH, map_location=self.device))
            self.classifier.to(self.device)
            self.classifier.eval()
        else:
            print(f"CẢNH BÁO: Không tìm thấy {EFFNET_PATH}. Chế độ sàng lọc sẽ bị tắt.")
            self.classifier = None

        # 2. Load YOLOv8
        print("   -> Loading YOLOv8m...")
        if os.path.exists(YOLO_PATH):
            self.detector = YOLO(YOLO_PATH)
        else:
            raise FileNotFoundError(f"Lỗi: Không tìm thấy model YOLO tại {YOLO_PATH}")

        print("Hệ thống sẵn sàng!")

    def preprocess_effnet(self, image_pil):
        img_np = np.array(image_pil)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (380, 380))
        img_normalized = img_resized.astype(np.float32) / 255.0

        # [H, W, C] -> [C, H, W] -> [1, C, H, W]
        img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)


ai_system = AI_System()

@app.get("/")
def home():
    return {"message": "AIDIMS AI Service is running", "version": MODEL_VERSION}


@app.post("/predict_findings")
async def predict_findings(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File ảnh lỗi: {str(e)}")

    results = {
        "model_version": MODEL_VERSION,
        "classification": {"status": "UNKNOWN", "confidence": 0.0},
        "findings": []
    }

    is_abnormal = True
    eff_score = 0.0

    if ai_system.classifier:
        img_tensor = ai_system.preprocess_effnet(img_pil)
        with torch.no_grad():
            output = ai_system.classifier(img_tensor).squeeze()
            eff_score = torch.sigmoid(output).item()

        results["classification"]["confidence"] = round(eff_score, 4)

        if eff_score < THRESHOLD_EFFNET:
            results["classification"]["status"] = "NORMAL"
            is_abnormal = False
            # Nếu bình thường, trả về luôn danh sách rỗng hoặc 1 object 'No finding'
            results["findings"].append({
                "label": "No finding",
                "confidence_score": 1.0 - eff_score,
                "bbox_xyxy": [0, 0, 0, 0]
            })
            return results
        else:
            results["classification"]["status"] = "ABNORMAL"

    if is_abnormal:
        yolo_results = ai_system.detector.predict(img_pil, conf=THRESHOLD_YOLO, verbose=False)
        detections = yolo_results[0]

        if len(detections.boxes) == 0:
            results["findings"].append({
                "label": "Suspicious (Unlocalized)",
                "confidence_score": eff_score,
                "bbox_xyxy": [0, 0, 0, 0]
            })
        else:
            for box in detections.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                class_name = detections.names[class_id]

                # Lấy tọa độ
                coords = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                results["findings"].append({
                    "label": class_name,
                    "confidence_score": round(score, 4),
                    "bbox_xyxy": [round(x, 2) for x in coords]
                })

    results["findings"].sort(key=lambda x: x['confidence_score'], reverse=True)

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

