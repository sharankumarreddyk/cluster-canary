---
name: model-card-writer
description: Generates Hugging Face style model cards with intended use, training data provenance, evaluation results sliced by subgroup, fairness considerations, limitations, and operational guidance. Use whenever a model is registered, promoted, released, or attached to a repo. A model without a card is not ship-ready.
---

# Model Card Writer

## When this skill applies

Trigger this skill when code:
- Promotes a model to `Production` in the MLflow registry.
- Tags a release (git tag, GitHub release) that includes model artifacts.
- Pushes a model to Hugging Face Hub, S3 release bucket, or model registry.
- Adds a `MODEL_CARD.md`, `model_card.md`, or similar file to a repo.

If a model is being shipped without a current model card, generate or update one before declaring done.

## The model card template

Save as `models/<model_name>/MODEL_CARD.md` and link from the README. Keep it under ~800 lines but never skip a section — write "N/A" with one sentence on why instead.

```markdown
# Model Card — <Model Name>

**Version:** v<n>  •  **Released:** <YYYY-MM-DD>  •  **License:** <SPDX>
**MLflow run:** <run_id>  •  **Git SHA:** <sha[:7]>
**Owner:** <name / team>  •  **Contact:** <email or Slack handle>

## 1. Intended use

- **Primary use case:** <one paragraph describing what the model predicts, who consumes the prediction, and what action it informs>
- **Primary intended users:** <e.g. an internal service, an end-user feature, an analyst dashboard>
- **Out-of-scope uses (do NOT use for):** <list 3–5 use cases the model was NOT evaluated for — e.g. high-stakes decisions, regulatory reporting, automated billing>

## 2. Training data

| Field | Value |
|---|---|
| Source | <dataset name + origin> |
| URL | <link to dataset> |
| Window covered | <start date – end date> |
| Rows after cleaning | <N> |
| License | <license> |
| Known biases | <e.g. geographic skew, temporal coverage gaps, sampling bias> |
| Sensitive attributes present? | <yes/no + which; direct PII vs. proxies> |
| Preprocessing summary | <key cleaning rules, feature transforms; explicitly call out target-component columns you dropped to prevent leakage> |

DVC reference: `data/processed/train_<version>.parquet @ <dvc-hash>`

## 3. Model details

| Field | Value |
|---|---|
| Architecture | <e.g. LightGBM regressor, num_leaves=63, n_estimators=800> |
| Inputs | <ordered feature list + dtypes; or schema link> |
| Output | <prediction column + range + post-processing> |
| Training time | <N> minutes on <hardware> |
| Hyperparameter search | <e.g. Optuna, 50 trials, optimized for temporal-holdout metric> |
| Framework versions | <pin major libs + Python> |
| Reproducibility | `mlflow run --version <sha>` or `dvc repro` |

## 4. Evaluation

### Overall (held-out test set)
| Metric | Value | Baseline |
|---|---|---|
| <primary metric> | <e.g. 2.18> | <baseline> |
| <secondary metric> | <…> | <baseline> |
| <calibration metric, if classification> | <…> | <baseline> |

### Temporal generalization (later window)
Primary-metric delta from in-distribution: +<X>%. Acceptable threshold: <Y>%.

### Sliced performance — REQUIRED
Report your primary metric per slice. Flag any slice with worst-metric > 1.5× overall.

| Slice | n | <metric> | Δ vs overall |
|---|---|---|---|
| <slice 1 — e.g. region A> | … | … | … |
| <slice 2> | … | … | … |
| <slice 3> | … | … | … |
| <slice 4> | … | … | … |
| <slice 5 — temporal / cohort bucket> | … | … | … |

### Performance on edge cases
| Scenario | Behavior |
|---|---|
| <out-of-distribution input> | <how the model degrades> |
| <degenerate input> | <safety bound applied> |
| <boundary case (e.g. NYE midnight, daylight saving switch)> | <variance / known issue> |

## 5. Operational behavior

- **Inference latency:** p50 <X>ms, p95 <Y>ms, p99 <Z>ms at batch=1 on <hardware>.
- **Throughput:** <RPS> sustained at p95 < <SLO> ms in load test (see `reports/loadtest.html`).
- **Memory footprint:** ~<M>MB resident in serving container.
- **Cold start:** Model loads in <T>s; warmup completes in <T2>s. Use `/readyz` to gate traffic.
- **Drift monitoring:** Evidently AI runs <cadence>; reference window = training data; thresholds in `src/<pkg>/monitoring/drift.py`.
- **Retraining trigger:** Auto-triggered when (data_drift_share > <T1>) OR (current_<metric> > <T2> × training) for 2 consecutive checks.

## 6. Limitations

- <Trained on a specific time window — call out untested periods.>
- <Population coverage — which segments are under-represented.>
- <Signal coverage — what's NOT in the features that real users might assume is.>
- <Target boundary — what the target does and doesn't include.>

## 7. Fairness considerations

- <List protected / sensitive attributes; whether they're features or excluded.>
- <Surface known proxies (zip code, device type, etc.) and how their slice metrics compare.>
- <Specify what the model should NOT be used for given these constraints.>

## 8. How to use this model

```python
import bentoml
runner = bentoml.<flavor>.get("<model_name>:latest").to_runner()
runner.init_local()
prediction = runner.predict.run(features_df)
```

Service-level API: see `src/<pkg>/serving/service.py` and the OpenAPI spec at `/docs`.

## 9. Citation

If you use this model card or methodology, cite the project repository.

## 10. Changelog

| Version | Date | Change | Trigger |
|---|---|---|---|
| v1 | <date> | Initial release | n/a |
```

## Filling it out — practical rules

1. **No N/A without one sentence of justification.** "N/A — model is regression, no fairness subgroups defined" is fine. Bare "N/A" is lazy.
2. **Numbers come from MLflow.** Don't retype them; copy from `mlflow.search_runs` output to avoid drift.
3. **The slice table is mandatory.** This is what makes the card credible to a senior reviewer. If you don't have slicing, fix the eval first then write the card.
4. **"Out-of-scope uses" is mandatory.** Tells the reader you understand the boundaries of your model.
5. **The changelog updates with every promotion.** Don't rewrite history; append.
6. **The "Owner / Contact" line is non-optional** — even for a personal project, put your name. Hiring managers like seeing ownership.

## Auto-generation helper

When generating a card programmatically, pull from MLflow and the DVC manifest:

```python
def render_model_card(run_id: str, output_path: Path) -> None:
    import mlflow, jinja2
    run = mlflow.get_run(run_id)
    template = jinja2.Template(Path(".claude/skills/model-card-writer/template.md.j2").read_text())
    output_path.write_text(template.render(
        run=run,
        params=run.data.params,
        metrics=run.data.metrics,
        tags=run.data.tags,
        slice_metrics=json.loads(mlflow.artifacts.load_text(f"runs:/{run_id}/slice_metrics.json")),
    ))
```

(Build `template.md.j2` from the markdown above; this skill can be referenced when generating it.)

## Anti-patterns to refuse

- A model card that's just a copy of the README. Different document, different audience.
- Eval results without sliced metrics — incomplete.
- "This model works well" without "this model fails when X" — every model has failure modes; surface them.
- Marketing language ("state-of-the-art", "best-in-class"). A model card is a spec sheet, not a brochure.
- Missing the changelog (people skip it; reviewers always check it).
