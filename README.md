# Compressible Dynamics Reproduction

This repository contains the compact experiment code and generated figures for the ECE 273 project on *Compressible Dynamics in Deep Overparameterized Low-Rank Learning and Adaptation*.

The repo is intentionally small:

- `experiments/matrix_completion.py`: synthetic deep matrix-completion reproduction.
- `experiments/deep_lora_stsb.py`: STS-B few-shot LoRA vs. Deep LoRA fine-tuning.
- `figs/`: generated figures used in the report.

LaTeX/report sources are not required to run the experiments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

For the STS-B experiment, install the additional model-training dependencies:

```bash
python -m pip install torch transformers datasets
```

## Matrix Completion

Run a quick smoke test:

```bash
python experiments/matrix_completion.py --quick
```

Run the default experiment used for the report:

```bash
python experiments/matrix_completion.py
```

This writes `matrix_completion.png`, `matrix_completion.pdf`, and `summary.txt` to `results/matrix_completion/`.

Default setting:

- Target matrix size `d = 200`
- Target rank `r = 3`
- Depth `L = 3`
- Observed-entry probability `0.20`
- `5000` GD steps over `5` seeds

Observed report-scale result:

| Method | Final relative masked loss | Mean runtime |
| --- | ---: | ---: |
| Full GD | `1.59e-05` | `3.95s` |
| Compressed GD, `gamma > 0` | `2.93e-06` | `0.34s` |
| Compressed GD, `gamma = 0` | `9.96e-01` | `0.33s` |

The basis-update version reached the same final-error scale as full GD with about an `11.7x` runtime speedup in the local NumPy implementation. The `gamma = 0` ablation fails because the observed-entry loss does not reveal the right subspace from the initial compressed basis alone.

This script uses `gamma = 10` after local tuning, so it should be viewed as an illustrative reproduction variant rather than the exact matrix-completion hyperparameter setting from the paper.

## STS-B Few-Shot Fine-Tuning

Run a small smoke test:

```bash
python experiments/deep_lora_stsb.py --samples 16 --seeds 0 --steps 5
```

Run the full few-shot sweep used for the report:

```bash
python experiments/deep_lora_stsb.py --samples 16 64 256 --seeds 0 1 2 --steps 500
```

This writes `stsb_results.csv`, `stsb_fewshot.png`, and `stsb_fewshot.pdf` to `results/stsb_fewshot/`. The experiment fine-tunes BERT-base-cased on small subsets of STS-B and reports Pearson correlation on the validation set. The script implements the LoRA adapters directly, so it does not require PEFT.

| Training examples | Vanilla LoRA | Deep LoRA | Gain |
| ---: | ---: | ---: | ---: |
| 16 | `0.570 +/- 0.021` | `0.668 +/- 0.049` | `+0.098` |
| 64 | `0.734 +/- 0.002` | `0.777 +/- 0.005` | `+0.043` |
| 256 | `0.813 +/- 0.007` | `0.833 +/- 0.003` | `+0.020` |

## Checked-In Figures

Generated figures used in the write-up are stored in `figs/`:

- `figs/application1_average_d200.png`
- `figs/application2_stsb_repro.pdf`

Rerunning scripts writes fresh outputs to `results/`; those outputs are ignored by Git so the repository stays clean.
