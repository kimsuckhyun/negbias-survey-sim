# Negative-Response Bias in LLM-Based Survey Simulation: Measurement and Calibration

Anonymous code release for the paper under review. This repository contains the synthetic-data pipeline, the LoRA
training code, the evaluation code, and the stored model outputs needed to reproduce the reported numbers without a GPU.

## Layout

| Folder | Content |
|---|---|
| `generation/` | Four-stage synthetic negative-response pipeline (`auto_survey_pipeline.py`: `gen-domains`, `gen-personas`, `gen-surveys`, `gen-responses`, `analyze`), the three-machine run script, the merge and verification script, and the script that groups 1 to 10 respondents into one training example |
| `prompts/` | Prompts used in the paper (domain, persona, questionnaire, response, panel, and the two instruction baselines) |
| `data/` | `train_neg_1to10_cohort.jsonl`: the 5,000 multi-respondent training examples (messages format) |
| `data_small/` | Domain pool (`domains.json`) and persona pool (`personas.json`) |
| `training/` | `train_lora_h100.py` (bf16 LoRA, 8 GPUs), `run_all_models.sh` (the seven models), `setup_env_h100.sh` |
| `evaluation/` | Scoring (`eval_multi_score.py`), dataset conversion, and the analysis scripts that produce the tables |
| `eval/` | Question files, human distributions, and stored Base and NT response distributions for OpinionQA, GOQA, GSS, and ISSP |

## Reproducing the tables without a GPU

Run from the repository root. Python 3.10+ with `numpy`, `pandas`, and `scipy`.

```bash
python3 evaluation/rep_wd.py                  # Rep and JS for every variant, alpha sweep, bootstrap
python3 evaluation/base_base_control.py       # Base+Base control for the mixture (appendix)
```

`base_base_control.py` compares the mixture of Base and NT with the mixture of Base and another Base model at the
same weight, and reports JS, Rep, and the negative-response bias with cluster-bootstrap intervals.

## Regenerating the data

1. Serve the teacher model (Gemma-4-31B-it, Q4_K_M) with llama.cpp and run `SEED=101 bash generation/run_gen5000.sh`
   on each machine (seeds 101, 102, 103). Set `ACL_BASE` to the output directory.
2. Merge and verify with `generation/merge_verify.sh` (edit the host names inside), which writes the deduplicated set
   and the balanced 15,000-response subset (7,500 strongly and 7,500 weakly negative).
3. Build the 5,000 training examples with `generation/resample_groups_1to10_persona_en.py`.

The merged set (23,112 responses) and the 15,000-response subset are too large for this repository and are not included;
the 5,000 training examples derived from the subset are in `data/`.

## Training

```bash
export ACL_ROOT=$PWD
bash training/setup_env_h100.sh
bash training/run_all_models.sh               # smoke test, then 1 epoch on 8 GPUs, for each of the seven models
```

LoRA rank 16, scaling 32, dropout 0.05, learning rate 2e-4, cosine decay with 5% warmup, 1 epoch, seed 42, effective
batch size 16 (313 steps for 5,000 examples). Adapter weights are not included.

## Notes

- The human response distributions in `eval/converted*` are aggregates derived from Pew Research Center's American
  Trends Panel (OpinionQA), GlobalOpinionQA, the General Social Survey, and ISSP. Please obtain and cite the original
  sources.
- The evaluation scripts expect this folder layout and use relative paths.
