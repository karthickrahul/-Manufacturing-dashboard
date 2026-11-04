"""Fine-tune a pretrained YOLO model on the prepared NEU-DET split.

Usage: python src/train_yolo.py --model yolo11n.pt --imgsz 320 --epochs 40 --name neu_n_320
Runs on CPU unless a GPU is visible (Ultralytics picks it automatically with --device "").
"""
import argparse
import json
import time

from ultralytics import YOLO

from common import DS, ROOT, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", required=True)
    ap.add_argument("--flipud", type=float, default=0.0)
    a = ap.parse_args()

    t0 = time.time()
    model = YOLO(a.model)
    model.train(
        data=str(DS / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        patience=a.patience, seed=SEED, deterministic=True, device=a.device, workers=2,
        flipud=a.flipud, project=str(ROOT / "runs" / "detect"), name=a.name, exist_ok=True,
        plots=True, verbose=False,
    )
    mins = (time.time() - t0) / 60
    (ROOT / "results").mkdir(exist_ok=True)
    meta = {"name": a.name, "model": a.model, "imgsz": a.imgsz, "epochs_requested": a.epochs,
            "batch": a.batch, "device": a.device, "flipud": a.flipud, "train_minutes": round(mins, 1)}
    (ROOT / "results" / f"train_{a.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {mins:.1f} min")


if __name__ == "__main__":
    main()
