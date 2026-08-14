.PHONY: setup data sheets evalset baseline styles verify report test app docker all

PY := .venv/bin/python
PIP := .venv/bin/pip

setup:
	python3.12 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt pytest

# Fetch COCO val2017 annotations, pick hard candidates, download those images.
data:
	mkdir -p data
	curl -sL -o data/ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
	cd data && unzip -o -q ann.zip "annotations/instances_val2017.json" "annotations/captions_val2017.json" && rm -f ann.zip
	$(PY) -m src.vlmhall.build_set
	$(PY) -m src.vlmhall.contact_sheets

# Rebuild data/eval_set.json from the hand-written verdicts in
# manual_verification.py. Verdicts are source, not generated.
evalset:
	$(PY) -m src.vlmhall.manual_verification

baseline:
	$(PY) -m src.vlmhall.evaluate

verify:
	$(PY) -m src.vlmhall.verify

report:
	$(PY) -m src.vlmhall.report

test:
	$(PY) -m pytest tests/ -q

app:
	.venv/bin/streamlit run app/demo.py

docker:
	docker build -t vlm-hallucination-eval .

all: data evalset baseline verify report test
