# Reproduction & Adversarial Audit: Chen et al. (Adv. Funct. Mater. 2026, 36:e75639)

**Target Study**: Chen et al., *"Active Learning-Accelerated Discovery of Fibrous Hydrogels with Tissue-Mimetic Viscoelasticity"*, *Advanced Functional Materials* 2026, 36:e75639.  
**Upstream Repository**: `Hydrogel_BO/`  
**Master Seed**: `20260919` (`env/seeds.yaml`)

---

## Executive Summary

This repository hosts a multi-agent reproduction, sensitivity stress-test, and adversarial audit of all computational, data, and quantitative claims in Chen et al. 2026. The audit is implemented under a dual-track architecture:
1. **Implementation Track (`src/`)**: Modular Python packages executing analysis tasks T00–T12, generating canonical datasets in `artifacts/canonical/`, Section 9 task cards in `artifacts/cards/`, and final synthesis in `report/`.
2. **Verification Track (`tests/`, `Makefile`)**: Automated 7-gate verification pipeline enforcing schema compliance, SHA-256 hash provenance, deterministic seed derivation, and quantitative assertions.

---

## Directory Layout

```text
audit/
├── Makefile                     # 7-gate verification and execution automation
├── README.md                    # Project documentation
├── PROJECT.md                   # Full audit specification and milestone matrix
├── env/
│   ├── seeds.yaml               # Master seed 20260919 and task seed allocations
│   ├── environment.yaml         # Conda environment definition
│   └── requirements.txt         # Pinned Python dependencies
├── artifacts/
│   ├── canonical/               # Ingested ground-truth tables (Gate 1 sign-off)
│   │   ├── points.parquet       # 168 experimental conditions (100 closed-loop + 68 Pareto)
│   │   ├── prior.parquet        # 43 literature prior conditions (Table S2, missing index 2)
│   │   ├── optima.parquet       # 7 reported optimal hydrogels with tolerance windows
│   │   ├── paper_claims.yaml    # Ground-truth constants and verbatim paper metrics
│   │   └── manifest.sha256      # SHA-256 cryptographic digests of canonical tables
│   └── cards/                   # Section 9 Draft-07 JSON task audit cards
│       ├── T00/card.json        # Data stewardship and canonical ingestion card
│       └── ...                  # Tasks T01 through T12
├── src/
│   ├── data/
│   │   └── ingest.py            # Canonical data parsing and reconciliation
│   ├── models/                  # GP surrogate and digital twin wrappers
│   ├── audit/                   # Audit execution modules (prospective, baselines, etc.)
│   └── utils/
│       ├── card_schema.json     # Section 9 Draft-07 JSON Schema
│       ├── card_validator.py    # Fast validation CLI using fastjsonschema
│       ├── hashing.py           # SHA-256 computation and manifest verification
│       └── seeds.py             # Deterministic seed derivation utility
├── tests/                       # Opaque-box automated test suite
└── report/                      # Final reports, claims ledger, and slides
```

---

## Environment Setup

The audit uses the Python virtual environment located at `Hydrogel_BO/venv/bin/python` linked to system site-packages:

```bash
# Verify environment dependencies
make check-env

# Ingest canonical data and issue Gate 1 sign-off (Task T00)
make data

# Verify canonical data manifest
make verify-canonical

# Validate Section 9 task cards
make verify-cards

# Verify provenance hashes
make verify-hashes

# Verify seed derivation protocol
make verify-seeds
```

---

## Verification Gates

Every milestone is gated by strict programmatic checks:
- **Gate 1**: Python environment and dependencies check (`make check-env`)
- **Gate 2**: Canonical data SHA-256 manifest verification (`make verify-canonical`)
- **Gate 3**: Section 9 JSON schema compliance (`make verify-cards`)
- **Gate 4**: Input/output artifact hash provenance verification (`make verify-hashes`)
- **Gate 5**: Deterministic seed derivation verification (`make verify-seeds`)
- **Gate 6**: Unit & integration test suite (`make verify-tests`)
- **Gate 7**: Synthesis and findings completeness (`make verify-report`)
