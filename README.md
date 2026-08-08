# 🚬 Cigarette Detection — YOLOv8

Real-time cigarette detection using a custom-trained YOLOv8 model, with a Streamlit app for image and video inference. Flags smoking in restricted/public areas with a visible violation alert.

**Author:** Nadeem Gohar

---

## Project structure

```
cigarette-detection/
├── Cigarette_Detection_YOLOv8_Training.ipynb   # Colab notebook: dataset → train → validate → export
├── app.py                                      # Streamlit inference app
├── requirements.txt
├── model/
│   └── best.pt                                 # trained weights (you generate this)
└── README.md
```

## 1. Train the model (Google Colab)

1. Upload `Cigarette_Detection_YOLOv8_Training.ipynb` to [Google Colab](https://colab.research.google.com).
2. `Runtime > Change runtime type > GPU (T4)`.
3. Get a free Roboflow API key: [app.roboflow.com](https://app.roboflow.com) → Settings → API Key.
4. Go to a cigarette dataset on Roboflow Universe (links are in the notebook's section 3), click **Download Dataset → YOLOv8 → show download code**, and paste that exact snippet into the notebook — this avoids guessing the wrong dataset version.
5. Run every cell top to bottom. Training runs `yolov8n` for up to 50 epochs (early stopping via `patience=15`).
6. Progress auto-saves to your Google Drive as it trains — if Colab disconnects, use the "resume" cell instead of restarting.
7. The last cell downloads `best.pt` to your computer.

**Want higher accuracy over speed?** Swap `YOLO("yolov8n.pt")` → `YOLO("yolov8s.pt")` in the training cell.

## 2. Run the app locally

```bash
# 1. Create the model folder and drop in your trained weights
mkdir model
mv ~/Downloads/best.pt model/best.pt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## 3. Deploy to Streamlit Community Cloud

1. Push this folder (including `model/best.pt`) to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to `app.py`.
3. Set Python version to **3.10** or **3.11** in the app's advanced settings.
4. Deploy.

## Features

- Image upload → bounding boxes + confidence scores + inference time
- Video upload → frame-by-frame detection with live preview + downloadable annotated output
- Red "smoking violation detected" banner whenever a cigarette is found, live during video playback too
- Adjustable confidence / IoU thresholds from the sidebar
- Dark glassmorphism UI

## Accuracy notes

- `yolov8n`: fastest, smallest (~3.2M params) — best for CPU-only Streamlit Cloud deployment.
- `yolov8s`: better mAP, still real-time, heavier — use if you have GPU inference available.
- If detections are noisy, raise the confidence threshold in the sidebar; if cigarettes are being missed, lower it.
