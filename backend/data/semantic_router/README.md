# Semantic router starter data

This directory contains an **AI-authored, non-production starter corpus** for developing IntoMath's multilingual embedding router. Every row is marked:

```json
"review_status": "needs_review"
```

No row should be treated as independently human-reviewed, production-ready, or as evidence of production accuracy. The files contain authored examples and selected repository fixture adaptations only; they do not contain collected user prompts.

## Current corpus

| Split | Rows | Semantic groups |
|---|---:|---:|
| Train | 144 | 48 |
| Validation | 36 | 12 |
| Test | 36 | 12 |
| **Total** | **216** | **72** |

Each semantic group has English, Vietnamese, and mixed English/Vietnamese variants. The corpus therefore has 72 rows for each language tag. Every problem type has 24 rows. Difficulty and visualization classes are intentionally less uniform because forcing every mathematical topic into an unnatural difficulty or visualization would reduce data quality.

The six `in_scope=false` rows are non-mathematical OOD examples used only to characterize abstention. They must not become category prototypes or training positives.

## Row format

Required fields:

- `id`
- `text`
- `language`
- `problem_type`
- `difficulty`
- `visualization_environment`
- `group_id`
- `source`
- `review_status`

Optional router-training metadata:

- `visualization_search_terms`: reviewed terms retrievable from semantically similar examples; these do not determine classification.
- `in_scope`: whether the row belongs to the supported mathematical distribution.
- `source_id`: stable provenance identifier when adapted from another checked-in source.

The runtime and validator use only Unicode NFKC and whitespace normalization. They do not rewrite semantic keywords.

## Leakage policy

Translations, paraphrases, typo variants, and OCR-like variants of one underlying prompt must share a `group_id` and remain in one split. `load_dataset` rejects duplicate IDs, exact normalized-text duplicates, conflicting group labels, cross-split groups, and cross-split `source_id` values.

The checked-in test split is locked for evaluation. Do not tune thresholds, select checkpoints, or edit model behavior against test outcomes. Validation is the only calibration split.

## Human-review workflow

Before using an example for a production decision:

1. A qualified reviewer checks the text, language tag, all three labels, group assignment, provenance, and any retrieval terms.
2. Corrections are applied consistently to every row in the group.
3. `review_status` is changed to `human_reviewed` only after that review.
4. Run the dataset validator and inspect the new class/language balance.
5. Recompute the data hash, base-model baseline, thresholds, and locked test report.

Fine-tuning is optional and must not be promoted merely because training completed. The evaluation report must compare it with the untouched base encoder and apply the documented promotion gates.
