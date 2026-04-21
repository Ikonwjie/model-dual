# BALM-MedualTime

Official codebase for the `model_dual.py` long-term time series forecasting experiments in this repository.

This repository is organized in a release-friendly way similar to [CALF](https://github.com/Hank0626/CALF): it includes a runnable training entrypoint, dependency list, shell scripts, dataset layout notes, and experiment summaries.

## Introduction

This project studies a dual-branch LLM-style forecasting model for multivariate time series forecasting.

The current public code path focuses on:

- `models/model_dual.py`: the main dual-branch forecasting model.
- `run_main.py`: the training and evaluation entrypoint.
- `data_provider/`: dataset loading and preprocessing.
- `utils/`: metrics, logger, time features, and training helpers.

The implementation uses:

- RevIN-based input normalization.
- patch-wise time series tokenization.
- pseudo-text token construction from GPT-2 word embeddings.
- a dual-branch GPT backbone with selective cross-branch bridge modules.
- P2T semantic injection and a forecasting readout head.

## Repository Structure

```text
.
├── checkpoints/
├── configs/
├── data_provider/
├── dataset/
│   └── ETT-small/
├── models/
│   └── model_dual.py
├── scripts/
│   └── long_term_forecasting/
├── utils/
├── requirements.txt
├── run.py
└── run_main.py
```

## Prerequisites

- Python 3.9 or newer is recommended.
- CUDA-capable GPU is recommended for training.
- Local GPT-2 weights must be available to `transformers`, because `models/model_dual.py` loads:

```python
GPT2Model.from_pretrained("gpt2", local_files_only=True)
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This repository currently uses the `ETT-small` datasets under:

```text
dataset/
└── ETT-small/
    ├── ETTh1.csv
    ├── ETTh2.csv
    ├── ETTm1.csv
    ├── ETTm2.csv
    └── weather.csv
```

For the current code, make sure at least the files below exist:

- `dataset/ETT-small/ETTh1.csv`
- `dataset/ETT-small/ETTh2.csv`
- `dataset/ETT-small/weather.csv`

## Training

The main entrypoint is:

```bash
python run.py
```

or equivalently:

```bash
python run_main.py
```

Example:

```bash
python run.py \
  --is_training 1 \
  --model_file ./models/model_dual.py \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --gpu 0
```

Predefined shell scripts are provided under:

```text
scripts/long_term_forecasting/
```

For example:

```bash
bash scripts/long_term_forecasting/ETTh1_pl96.sh
```

## Checkpoints and Logs

Training outputs are saved under:

```text
checkpoints/{time}_{model}_{data}_ft{features}_sl{seq_len}_pl{pred_len}_{itr}
```

Examples:

- `1555_model_dual_ETTh1_ftM_sl96_pl96_0`
- `1603_model_dual_ETTh1_ftM_sl96_pl192_0`

Each run directory typically contains:

- `*_record_s{seed}.log`: full training log.
- `checkpoint`: best saved model state.
- `checkpoint.progress`: saved progress value.
- `result_s{seed}.txt`: final test metrics.

## Current ETTh1 Results

Using `model_dual.py`, `seq_len=96`, `features=M`, `seed=43`:

| pred_len | best vali_mse | test MSE | test MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|---:|
| 96  | 0.700918 | 0.367320 | 0.390966 | 0.606069 | 9.374969 |
| 192 | 1.000555 | 0.421068 | 0.424540 | 0.648898 | 9.503657 |
| 338 | 1.304719 | 0.457881 | 0.445820 | 0.676669 | 9.720716 |
| 720 | 1.597509 | 0.462477 | 0.469691 | 0.680057 | 10.518018 |

## Saved Experiment Notes

The repository currently includes archived experiment notes:

- `configs/1501_namespace.txt`
- `configs/1501_summary.txt`

These files record the full argument namespace and summarized results for the `1501` baseline experiment.

## Citation

If this repository helps your work, cite your corresponding paper or project report here after finalizing the publication metadata.

```bibtex
@misc{balm_medualtime,
  title  = {BALM-MedualTime},
  author = {Your Name},
  year   = {2026},
  note   = {Code release}
}
```

## Acknowledgements

This repository builds on ideas and tooling patterns from open-source time series forecasting projects, including:

- CALF
- Autoformer
- PatchTST
- Time-Series-Library

