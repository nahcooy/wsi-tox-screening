.PHONY: setup backend frontend dev test

setup:
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r backend/requirements.txt

backend:
	.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend

frontend:
	cd frontend && npm install && npm run dev

dev:
	@echo "Run 'make backend' and, when npm is available, 'make frontend' in separate terminals."

test:
	cd backend && ../.venv/bin/python -m pytest tests
