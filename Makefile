.ONESHELL:
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# AltiMap dev orchestration -- Django + SQLite backend, React/Vite frontend.
# `viewer/` (the DA3 pipeline itself) is untouched; this just wires up and
# runs the backend/frontend around it. See CLAUDE.md for the architecture.

BACKEND_DIR    := backend
FRONTEND_DIR   := frontend
BACKEND_VENV   := $(BACKEND_DIR)/.venv
BACKEND_PY     := $(BACKEND_VENV)/bin/python
BACKEND_PORT   := 8000
FRONTEND_PORT  := 5173
RUNDIR         := .run

.PHONY: help install install-backend install-frontend migrate seed \
        backend frontend dev yolo stop restart status logs clean

help:
	@echo "AltiMap -- available targets:"
	@echo "  make yolo             install deps, migrate + seed the db, start backend & frontend, verify both are up"
	@echo "  make install          install backend (uv) and frontend (npm) dependencies"
	@echo "  make migrate          run Django migrations"
	@echo "  make seed             seed the db with demo scenes (skips if scenes already exist; --force via 'make seed FORCE=1')"
	@echo "  make dev              start backend + frontend in the background, wait until both respond"
	@echo "  make backend          run the Django dev server in the foreground"
	@echo "  make frontend         run the Vite dev server in the foreground"
	@echo "  make status           show whether backend/frontend are up"
	@echo "  make logs             tail backend + frontend logs"
	@echo "  make stop             stop backend + frontend started by 'make dev'/'make yolo'"
	@echo "  make restart          stop, then dev"
	@echo "  make clean            DESTRUCTIVE: remove venv, node_modules, sqlite db and media"

# ---- install --------------------------------------------------------------

install: install-backend install-frontend

install-backend:
	@echo "==> backend: uv venv + pip install (this pulls torch + depth_anything_3 the first time -- several GB)"
	test -d $(BACKEND_VENV) || uv venv --python 3.12 $(BACKEND_VENV)
	uv pip install --python $(BACKEND_PY) -r $(BACKEND_DIR)/requirements.txt

install-frontend:
	@echo "==> frontend: npm install"
	cd $(FRONTEND_DIR) && npm install

# ---- database ---------------------------------------------------------

migrate:
	@echo "==> backend: migrate"
	$(BACKEND_PY) $(BACKEND_DIR)/manage.py migrate

seed: migrate
	@echo "==> backend: seed_scenes"
	$(BACKEND_PY) $(BACKEND_DIR)/manage.py seed_scenes $(if $(FORCE),--force,)

# ---- run --------------------------------------------------------------

backend:
	$(BACKEND_PY) $(BACKEND_DIR)/manage.py runserver 127.0.0.1:$(BACKEND_PORT)

frontend:
	cd $(FRONTEND_DIR) && npm run dev -- --host 127.0.0.1 --port $(FRONTEND_PORT)

dev: stop
	@mkdir -p $(RUNDIR)
	echo "==> starting backend on :$(BACKEND_PORT)"
	nohup $(BACKEND_PY) $(BACKEND_DIR)/manage.py runserver 127.0.0.1:$(BACKEND_PORT) \
		> $(RUNDIR)/backend.log 2>&1 &
	echo $$! > $(RUNDIR)/backend.pid

	echo "==> starting frontend on :$(FRONTEND_PORT)"
	nohup npm --prefix $(FRONTEND_DIR) run dev -- --host 127.0.0.1 --port $(FRONTEND_PORT) \
		> $(RUNDIR)/frontend.log 2>&1 &
	echo $$! > $(RUNDIR)/frontend.pid

	$(MAKE) --no-print-directory _wait-healthy
	@echo ""
	@echo "Backend:  http://127.0.0.1:$(BACKEND_PORT)  (admin at /admin, api at /api/scenes/)"
	@echo "Frontend: http://127.0.0.1:$(FRONTEND_PORT)"
	@echo "Logs:     make logs   |   Stop: make stop"

yolo: install migrate seed dev
	@echo ""
	@echo "Everything is up. Open http://127.0.0.1:$(FRONTEND_PORT)"

_wait-healthy:
	@echo "==> waiting for backend..."
	for i in $$(seq 1 60); do \
		curl -sf http://127.0.0.1:$(BACKEND_PORT)/health >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	if ! curl -sf http://127.0.0.1:$(BACKEND_PORT)/health >/dev/null 2>&1; then \
		echo "backend did not come up -- see $(RUNDIR)/backend.log"; tail -n 40 $(RUNDIR)/backend.log; exit 1; \
	fi
	@echo "    backend OK"
	@echo "==> waiting for frontend..."
	for i in $$(seq 1 60); do \
		curl -sf http://127.0.0.1:$(FRONTEND_PORT)/ >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	if ! curl -sf http://127.0.0.1:$(FRONTEND_PORT)/ >/dev/null 2>&1; then \
		echo "frontend did not come up -- see $(RUNDIR)/frontend.log"; tail -n 40 $(RUNDIR)/frontend.log; exit 1; \
	fi
	@echo "    frontend OK"

status:
	@echo -n "backend  (:$(BACKEND_PORT)):  "
	@curl -sf http://127.0.0.1:$(BACKEND_PORT)/health >/dev/null 2>&1 && echo up || echo down
	@echo -n "frontend (:$(FRONTEND_PORT)):  "
	@curl -sf http://127.0.0.1:$(FRONTEND_PORT)/ >/dev/null 2>&1 && echo up || echo down

logs:
	@touch $(RUNDIR)/backend.log $(RUNDIR)/frontend.log
	tail -n 40 -f $(RUNDIR)/backend.log $(RUNDIR)/frontend.log

stop:
	@echo "==> stopping backend + frontend"
	@if [ -f $(RUNDIR)/backend.pid ]; then kill $$(cat $(RUNDIR)/backend.pid) 2>/dev/null || true; rm -f $(RUNDIR)/backend.pid; fi
	@if [ -f $(RUNDIR)/frontend.pid ]; then kill $$(cat $(RUNDIR)/frontend.pid) 2>/dev/null || true; rm -f $(RUNDIR)/frontend.pid; fi
	@# Belt and braces: also free the exact ports, in case a stale/untracked
	@# process (e.g. a manually-started runserver from an earlier session) is
	@# already sitting on them -- this only ever targets these two literal
	@# ports, never processes by name.
	@fuser -k -TERM $(BACKEND_PORT)/tcp 2>/dev/null || true
	@fuser -k -TERM $(FRONTEND_PORT)/tcp 2>/dev/null || true
	@sleep 0.3

restart: stop dev

# ---- destructive ------------------------------------------------------

clean: stop
	@echo "This removes $(BACKEND_VENV), $(FRONTEND_DIR)/node_modules, the sqlite db and all media."
	@echo "Re-run as: make clean CONFIRM=1"
	@if [ "$(CONFIRM)" != "1" ]; then echo "aborted -- nothing removed"; exit 1; fi
	rm -rf $(BACKEND_VENV) $(FRONTEND_DIR)/node_modules $(BACKEND_DIR)/db.sqlite3 $(BACKEND_DIR)/media $(RUNDIR)
