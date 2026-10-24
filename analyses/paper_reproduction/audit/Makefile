# Makefile for Adversarial Audit Toolchain
# Chen et al. (Adv. Funct. Mater. 2026, 36:e75639) Reproduction & Audit
SHELL := /bin/bash
PYTHON := /Users/frederick/Visual Studio /Material discovery/Hydrogel_BO/venv/bin/python
export PYTHONPATH := /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages:$(CURDIR)/src:$(CURDIR)/tests:$(PYTHONPATH)
export MPLCONFIGDIR := $(CURDIR)/.mplconfig

.PHONY: all help check-env data verify-canonical verify-cards verify-hashes verify-seeds test verify-tests verify clean

all: help

help:
	@echo "======================================================================"
	@echo "Chen et al. 2026 Reproduction & Adversarial Audit Verification Suite"
	@echo "======================================================================"
	@echo "  make check-env        Check Python environment & required dependencies"
	@echo "  make data             Ingest raw data and generate canonical tables (T00)"
	@echo "  make verify-canonical Verify SHA-256 digests of canonical parquet tables"
	@echo "  make verify-cards     Validate all Section 9 Draft-07 task cards"
	@echo "  make verify-hashes    Verify provenance SHA-256 hashes inside cards"
	@echo "  make verify-seeds     Verify deterministic task seed derivation"
	@echo "  make verify           Execute full Gate 1 verification suite"
	@echo "  make clean            Clean temporary files and cache directories"

check-env:
	@echo "=== Gate 1.1: Checking Environment & Dependencies ==="
	@"$(PYTHON)" -c "import torch, botorch, gpytorch, sklearn, scipy, pandas, numpy, pyarrow, fastparquet, yaml, fastjsonschema, pytest; print('✅ All core packages verified!')"

data: check-env
	@echo "=== Gate 1.2: Ingesting Canonical Data (T00) ==="
	@"$(PYTHON)" src/data/ingest.py

verify-canonical:
	@echo "=== Gate 2: Verifying Canonical Data SHA-256 Manifest ==="
	@"$(PYTHON)" src/utils/hashing.py --verify artifacts/canonical/manifest.sha256 --base-dir .

verify-cards:
	@echo "=== Gate 3: Validating Section 9 Card Schemas ==="
	@"$(PYTHON)" src/utils/card_validator.py --cards-dir artifacts/cards --schema src/utils/card_schema.json

verify-hashes:
	@echo "=== Gate 4: Verifying Card Provenance SHA-256 Hashes ==="
	@"$(PYTHON)" src/utils/card_validator.py --cards-dir artifacts/cards --verify-hashes --base-dir .

verify-seeds:
	@echo "=== Gate 5: Verifying Deterministic Seed Derivation ==="
	@"$(PYTHON)" src/utils/seeds.py --verify --config env/seeds.yaml

test:
	@"$(PYTHON)" -m pytest tests/ -v

verify-tests:
	@echo "=== Gate 6: Running Pytest Test Suite ==="
	@"$(PYTHON)" -m pytest tests/ -v

verify: check-env verify-canonical verify-cards verify-hashes verify-seeds verify-tests
	@echo ""
	@echo "======================================================================"
	@echo "✅ ALL GATE 1 INGESTION & INFRASTRUCTURE CHECKS PASSED SUCCESSFULLY"
	@echo "======================================================================"

clean:
	@rm -rf .mplconfig __pycache__ src/**/__pycache__
	@echo "Cleaned cache directories."
