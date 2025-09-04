# NucWorm Benchmark

This repository contains scripts and results for the NucWorm benchmark. See `scripts/` for method pipelines and `utils/` for evaluation tools.

## Repository structure

```
docs/                      Documentation and notes
outputs/                   Large outputs (ignored by git)
scripts/                   Pipelines for methods and data utilities
slurm/                     Submission wrappers to orchestrate jobs
```

Quickly submit center extraction arrays for supported methods:

```bash
bash slurm/submit_center_extraction.sh --methods cellpose_sam,cellpose3
```

## Neuron Detection (d<sub>th</sub> = 3 μm)

### nnUNet Performance

| Dataset | F1 | Precision | Recall |
|---------|----|-----------|---------|
| **NEJATBAKHSH20** | **0.856±0.041** | **0.939±0.052** | **0.788±0.053** |
| **WEN20** | **0.783±0.032** | **0.855±0.058** | **0.726±0.041** |
| **YEMINI21** | **0.884±0.019** | **0.947±0.037** | **0.830±0.019** |
| **Overall** | **0.847±0.050** | **0.922±0.062** | **0.785±0.057** |

TODO: try more preprocessing then rerun cellpose methods below....

### CellPose+SAM Performance (flow threshold = 0.4)

| Dataset | F1 | Precision | Recall |
|---------|----|-----------|---------|
| **NEJATBAKHSH20** | **0.602** | **0.890** | **0.455** |
| **WEN20** | **0.562** | **0.477** | **0.685** |
| **YEMINI21** | **0.609** | **0.502** | **0.776** |
| **Overall** | **0.632** | **0.661** | **0.659** |

### CellPose3 Performance (flow threshold = 0.4)

| Dataset | F1 | Precision | Recall |
|---------|----|-----------|---------|
| **NEJATBAKHSH20** | **0.500** | **0.400** | **0.758** |
| **WEN20** | **0.158** | **0.088** | **0.846** |
| **YEMINI21** | **0.114** | **0.061** | **0.889** |
| **Overall** | **0.326** | **0.245** | **0.811** |

### CellPose3+Denoising Performance (flow threshold = 0.4)

| Dataset | F1 | Precision | Recall |
|---------|----|-----------|---------|
| **NEJATBAKHSH20** | **0.627** | **0.669** | **0.590** |
| **WEN20** | **0.105** | **0.056** | **0.859** |
| **YEMINI21** | **0.117** | **0.063** | **0.844** |
| **Overall** | **0.423** | **0.379** | **0.739** |

