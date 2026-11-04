# End-to-end pipeline. Run `make all` (CPU is fine; ~1 h with the default nano model).
PY ?= python3
WEIGHTS ?= runs/detect/neu_n_224/weights/best.pt
IMGSZ ?= 224
RECALL_TARGET ?= 0.98

.PHONY: all data prepare baseline train val test delay stream impact dashboard static test-units clean-results

all: data prepare baseline train val test delay stream impact

data:
	bash scripts/get_data.sh

prepare: data
	$(PY) src/prepare_data.py

baseline: prepare
	$(PY) src/baseline_svm.py

train: prepare
	$(PY) src/train_yolo.py --model yolo11n.pt --imgsz $(IMGSZ) --epochs 40 --name neu_n_$(IMGSZ)

# Step 1: pick the confidence threshold on VAL only: highest threshold with part-level flag recall >= target
val:
	$(PY) src/evaluate.py --weights $(WEIGHTS) --split val --imgsz $(IMGSZ) --recall-target $(RECALL_TARGET) --metric flag_recall

# Step 2: score TEST once at the frozen threshold
test:
	$(PY) src/evaluate.py --weights $(WEIGHTS) --split test --imgsz $(IMGSZ)

delay:
	$(PY) src/detection_delay.py --runs 3000

stream:
	$(PY) src/make_stream.py

impact:
	$(PY) src/impact_model.py

dashboard:
	streamlit run app/dashboard.py

# Static copy of the dashboard for hosts without a Python server (Netlify, GitHub Pages): web/index.html
static:
	$(PY) scripts/build_static.py

test-units:
	$(PY) -m pytest -q tests
