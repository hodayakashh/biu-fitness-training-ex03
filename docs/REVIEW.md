# Code Review — BIU DRL Ex03 (`fitness-rl`)

TA review against Dr. Yoram Segal's professional-software guidelines (V3.00) and
the Ex03 assignment spec. Conducted with 6 parallel review agents, followed by
direct remediation of the critical findings.

**Branch reviewed:** `worktree-quality-review` (off `main` @ `b41c99f`).

## Summary scorecard

| # | Category (Dr. Segal §) | Status | Notes |
|---|------------------------|:------:|-------|
| 1 | Documentation & structure (§19,§24) | ✅ | PRDs/PLAN/TODO/README aligned to code (state_dim, reward formulas, module map) |
| 2 | SDK architecture (§3) | ✅ | `FitnessRLSDK` is the sole entry point; notebook uses it exclusively |
| 3 | OOP / no duplication (§4) | ⚠️ | REINFORCE/A2C share env + `_SharedTrunk`; rollout boilerplate could share a base class |
| 4 | **API Gatekeeper (§5)** | ✅ | **Fixed:** Kaggle download now routed through `ApiGatekeeper`; service-specific limits applied |
| 5 | Config-driven, no hardcoding (§11) | ✅ | λ₁/λ₂, γ, dims, thresholds all from `setup.json`; minor `.get()` fallback duplication remains |
| 6 | File size ≤150 lines (§6) | ✅ | Largest non-blank/non-comment file = 132 lines |
| 7 | TDD & coverage ≥85% (§8,§9) | ✅ | **100% coverage**, 133 tests, error paths covered |
| 8 | Zero Ruff violations (§10) | ✅ | `ruff check` + `ruff format --check` both clean |
| 9 | Secrets & security (§12) | ✅ | No secrets in source; `.env-example` dummy-only; `*.pt/*.pth` now git-ignored |
| 10 | `uv` only (§13) | ✅ | All tooling via `uv run`; CI uses `uv` exclusively |
| 11 | Version tracking (§14) | ✅ | `version.py` 1.00 ↔ `setup.json` ↔ `rate_limits.json`, all validated at startup |
| 12 | **Quality automation (CI/CD, hooks)** | ✅ | **Added:** GitHub Actions, pre-commit hooks, Makefile |
| 13 | **Version management / git history** | ⚠️ | Messages are strong on *why*; past bulk commits + no branches/tags. Convention adopted going forward (this branch + small commits) |

Legend: ✅ meets requirement · ⚠️ acceptable with recommendation · ❌ blocking (none remain).

## Findings by agent

### Agent 1 — Version management
- History (9 commits) tells a phased story with unusually good *why*-focused messages,
  but two **bulk commits** mix concerns: `78023cb` (20 files: data pipeline + `uv.lock` +
  scaffold tests + a committed `.coverage` artifact) and `f41ba7c` (plotter + notebook +
  README **and a hidden reward-function change**). No feature branches, no version tags.
- **Recommendation (going forward):** short-lived `feat/*` branches → PR → squash-merge;
  Conventional Commits with domain scopes (`lstm`, `reinforce`, `a2c`, `data`, `env`,
  `gatekeeper`); annotated tag `v1.00` mirroring `version.py`. Past history is left intact
  (rewriting published history needs a force-push and is out of scope for this review).

### Agent 2 — Quality standards & CI/CD  → fixed
- Added `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `Makefile`; added
  `pre-commit` to dev dependencies. pyproject quality gates (`fail_under=85`, ruff rules)
  were already present.

### Agent 3 — Architecture & code  → critical items fixed
- **❌→✅ Gatekeeper was dead code:** `ApiGatekeeper` existed but no Kaggle call went
  through it, and it loaded only the `default` rate limits, silently ignoring the stricter
  `kaggle` block (10 rpm). Fixed (see Remediation).
- ⚠️ Trainer rollout duplication and a few `.get()` fallback literals remain (non-blocking).
- ✅ Penalty weights λ₁/λ₂ are correctly config-driven (not hardcoded).

### Agent 4 — Documentation  → fixed
- Corrected `state_dim` 4→5 across PRDs/PLAN, stale reward formulas, an incomplete module
  map, stale TODO/PRD statuses, and bare `pip`/`kaggle` in README (now `uv run`).

### Agent 5 — Security & configuration  → fixed
- Added generic `*.pt`/`*.pth` to `.gitignore`. No secrets in source; configs versioned;
  startup version validation confirmed working.

### Agent 6 — Tests & coverage  → fixed
- Added SDK tests + trainer/env/plotter/gatekeeper edge-case tests, taking coverage
  **39%→100% on `sdk.py`** and **95.99%→100% overall**.

## Remediation applied in this branch

1. **`fix(gatekeeper)`** — `ApiGatekeeper` now takes a `service` argument and applies that
   service's limits (`kaggle`: 10 rpm) with `default` fallback, and validates the
   `rate_limits.json` version against the code version at construction.
2. **`feat(data)`** — `DataService.download_dataset()` (exposed via `FitnessRLSDK`) downloads
   the configured Kaggle dataset **through `gatekeeper.execute(...)`**, satisfying the
   "all external API calls go through the gatekeeper" red-line. The Kaggle client and
   gatekeeper are dependency-injected so tests never touch the network/credentials.
3. **Tooling** — CI, pre-commit, Makefile, `*.pt`/`*.pth` ignore, full `ruff format`.
4. **Docs** — PRDs/PLAN/TODO/README aligned to the implemented code.

## CI/CD — what it does and how to verify

`.github/workflows/ci.yml` runs on every `push` and `pull_request`, on a Python
3.10/3.11/3.12 matrix:

1. install `uv` (`astral-sh/setup-uv`)
2. `uv sync --all-extras`
3. `uv run ruff check src/ tests/`
4. `uv run ruff format --check src/ tests/`
5. `uv run pytest --cov=src/fitness_rl --cov-report=term-missing` (fails if coverage < 85%)

Verify locally with the Makefile (mirrors CI):

```bash
make install      # uv sync --all-extras
make lint         # uv run ruff check src/ tests/
make test         # uv run pytest  → 133 passed, 100% coverage
uv run ruff format --check src/ tests/   # format gate
uv run pre-commit run --all-files        # same hooks CI/commits enforce
```

`.pre-commit-config.yaml` runs `ruff-check --fix` + `ruff-format` (and whitespace/EOF/YAML
hooks) on every commit, and the full `pytest` suite on `pre-push`. Install once with
`uv run pre-commit install`.

## Final quality gates (this branch)

- `ruff check src/ tests/` → **All checks passed**
- `ruff format --check src/ tests/` → **50 files already formatted**
- `pytest` → **133 passed**, **TOTAL coverage 100.00%** (gate 85%)

---

## v1.01 — Instructor-style theoretical review & remediation (2026-06-13)

A second review took the lecturer's perspective, scoring the project against Dr. Segal's
five emphasised points (LSTM↔RL distinction, balanced reward, REINFORCE↔A2C analysis,
practical interpretation, preprocessing) and the three "corner" traps (data leakage, time
encoding, hardcoding). The engineering and the REINFORCE↔A2C analysis scored highly; four
findings were actioned (see PLAN.md **ADR-001** for the design rationale).

| # | Finding | Severity | Resolution |
|---|---------|:--------:|-----------|
| 1 | **LSTM barely action-dependent → partly degenerate MDP.** Action/state collinearity in the K-Means labels meant the agent's choice had little causal grip on the predicted next state. | 🔴 | Hybrid world model (ADR-001): `muscle_balance_score` is now governed by the chosen action's empirical muscle profile, so actions causally and persistently change the state. |
| 2 | **Inert `imbalance_penalty`.** It read an LSTM dimension that could not be made action-aware; the real variety driver was an out-of-model `repetition_penalty` hack. | 🔴 | `imbalance_penalty` is now action-aware *through the environment* (ADR-001). Empirically it is not sufficient alone, so `repetition_penalty` is weighted **co-equally** (λ₂ = λ₃ = 1.5); both are needed — see report §6. |
| 3 | **Action names not anchored to clusters.** `ACTION_LABELS` mapped arbitrary K-Means IDs to fixed names; the report's per-action narrative could be mislabelled. | 🟠 | `DataPreprocessor.describe_clusters` derives labels from each cluster's dominant muscle group + load tier; plots/report use these data-driven labels. |
| 4 | **Default/config divergence.** `reward.py` default `lambda_repetition=0.5` ≠ `setup.json` `2.0`. | 🟡 | Code defaults aligned with `setup.json`; λ values re-balanced and kept in sync. |

**Honest scope note.** The LSTM is intentionally no longer the *whole* world model: load/duration/
temporal dynamics remain learned, muscle-balance dynamics are now analytic and action-conditioned.
This "known + learned dynamics" decomposition is standard in model-based RL and is documented as a
deliberate trade-off rather than hidden.
