PY := .venv/bin/python
ENV := set -a; [ -f .env ] && . ./.env; set +a;

.PHONY: setup corpus index ask serve test eval-retrieval eval-answers questions

setup:            ## Python environment
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

corpus:           ## Download current law texts from Finlex and split them into sections
	$(PY) -m lawqa.fetch && $(PY) -m lawqa.parse

index:            ## Keyword (stem + 5-gram) and embedding indexes (~$0.01)
	$(ENV) $(PY) -m lawqa.index

serve:            ## Chat page and API on http://localhost:8200
	$(ENV) .venv/bin/uvicorn lawqa.api:app --port 8200

test:             ## Offline tests with a fake model (no key needed)
	$(PY) -m pytest -q tests

eval-retrieval:   ## Search modes on dev, test and hand-written questions
	$(ENV) for s in dev test handwritten; do $(PY) -m eval.retrieval --split $$s; done

eval-answers:     ## End-to-end answers, judged (~$3 with the Sonnet checker)
	$(ENV) $(PY) -m eval.answers --pipelines rag,agent

questions:        ## Regenerate the generated question set (~$0.35)
	$(ENV) $(PY) -m eval.make_questions --n 150
