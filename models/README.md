# Model weights directory

Place model weight files here:

```
models/
├── yolo11_book_detector.pt   ← [PHASE 3] Custom fine-tuned book detector
│                                 If absent, pretrained YOLO11n (COCO) is used
│                                 and a clear info message is logged.
│                                 Train your own with: ultralytics train ...
├── yolo/
│   └── yolo11n.pt            # Auto-downloaded by ultralytics on first run
│                             # (COCO pretrained, class 73 = 'book')
├── trocr/                    # Auto-downloaded by HuggingFace transformers
├── segformer/                # Auto-downloaded by HuggingFace transformers
├── dqn/
│   └── dqn_checkpoint.pth   # Generated during Phase 5 training
└── robot/
    └── desk_arm.urdf         # Virtual robot URDF — generated in Phase 7
```

## YOLO11 Custom Book Detector (Phase 3)

To use a custom-trained model instead of the pretrained COCO fallback:

1. Train YOLO11 on a book dataset:
   ```bash
   yolo train model=yolo11n.pt data=books.yaml epochs=50 imgsz=640
   ```

2. Copy `runs/detect/train/weights/best.pt` to:
   ```
   models/yolo11_book_detector.pt
   ```

3. Restart the dashboard — `BookDetector` will automatically prefer the custom model.

**If `yolo11_book_detector.pt` does not exist:**
- The detector logs an info message (not an error).
- It automatically falls back to pretrained `yolo11n.pt` (COCO class 73 = book).
- The project continues to work normally — no crash, no manual intervention needed.
