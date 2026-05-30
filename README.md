# ML-Based Boundary Treatment for MPS

## Neural Network Architecture

![Network Architecture](figs/network_architecture.png)

Inference-only workflow for the trained neural networks that predict the
ghost-particle contribution to MPS differential operators. Training is **not**
included: the trained `.h5` models are loaded, run on the test data, and their
predictions are compared against ground truth.

The five operators (referred to below as `<operator>`) are:

```
n0 | gradientScalar | laplacianScalar | divergenceVector | laplacianVector
```

---

## 1. What's in the source

### Code

| File | What it is / generates |
|------|------------------------|
| `common_pipeline.py` | Shared preprocessing and metrics used by both scripts (CSV loading, feature engineering, column selection, CNN-group ordering, WG reconstruction, error metrics). Imported as a library — produces nothing on its own. No TensorFlow dependency. |
| `1_save_normalization.py` | **Run once.** Reads the training data a single time and computes the normalization coefficients (per-feature mean / scale) for **all five operators**. Generates the `normalization_weights/` folder. Needs only `numpy`, `pandas`, `scikit-learn`. |
| `2_predict_and_compare.py` | **Inference.** Loads the test data + the saved coefficients + the trained `.h5` model for one operator, predicts, and compares against ground truth. Generates the `output/<operator>/` folder. Needs `numpy`, `pandas`, `matplotlib`, `tensorflow`. |

### Data (`data/`)

| File | What it is |
|------|------------|
| `dataForTrainingAndValidation.tar.xz` | Compressed training+validation data. **Must be extracted to `dataForTrainingAndValidation.csv` before step 1.** Used only to fit the normalization coefficients. |
| `test_near_wall.csv` | Near-wall test particles — the points that are actually predicted and scored. |
| `test_not_near_wall.csv` | Background (non-near-wall) points, used only as context in the spatial plots. |
| `feature_description.md` | Reference description of the input/output columns. |

### Models (`model/`)

Trained Keras networks, one **pair** per operator: the full model `<stem>.h5`
(structure) and `<stem>_weights.h5` (weights). The stem is derived from the
operator's output columns:

| `<operator>` | model stem in `model/` |
|--------------|------------------------|
| `n0`               | `n0OG` |
| `gradientScalar`   | `gradientScalarOGAll` |
| `laplacianScalar`  | `laplacianScalarOG` |
| `divergenceVector` | `divergenceVectorOG` |
| `laplacianVector`  | `laplacianVectorOGAll` |

(If a file is named differently — e.g. a renamed variant like
`gradientScalarOGAllMJ` — set `MODEL_BASENAME` in `2_predict_and_compare.py`.)

### Other

| File | What it is |
|------|------------|
| `README.md` | This file. |
| `LICENSE` | License. |

---

## 2. How to run

```bash
# 1. Extract the training data (one time)
cd data
tar -xf dataForTrainingAndValidation.tar.xz      # -> dataForTrainingAndValidation.csv
cd ..

# 2. Compute normalization coefficients for all five operators (run once)
python 1_save_normalization.py

# 3. Run inference + comparison for one operator
#    Set `name` at the top of the script to the operator you want, then:
python 2_predict_and_compare.py
```

Both scripts anchor every path to their own location, so they can be launched
from any working directory. `name`, `OG`, `psum`, and `outputNormalization` must
match between the two scripts.

---

## 3. What gets created

Running the scripts produces two folders.

### `normalization_weights/`  (created by step 1)

One coefficient file per operator — the per-feature `mean` / `scale` for the
wide input and for each CNN stencil group:

```
normalization_weights/
├── n0OG_norm.pkl
├── gradientScalarOGAll_norm.pkl
├── laplacianScalarOG_norm.pkl
├── divergenceVectorOG_norm.pkl
└── laplacianVectorOGAll_norm.pkl
```

Step 2 reads the file matching its configured operator; the training data is
never touched again.

### `output/<operator>/`  (created by step 2)

One subfolder per operator (e.g. `output/laplacianScalar/`), holding the
comparison results. When `OG = "on"`, two rounds are written — the raw
ghost-contribution (`*_Gcont.*`) and the total field after adding the `WG`
baseline back (unsuffixed):

```
output/
└── <operator>/
    ├── <stem>.svg            / <stem>_Gcont.svg            spatial ground-truth vs predicted
    ├── <stem>_diff.svg       / <stem>_diff_Gcont.svg       difference map
    ├── <stem>_PVSGT.svg      / <stem>_PVSGT_Gcont.svg      predicted-vs-ground-truth scatter
    ├── <stem>_HISTOGRAM.svg  / <stem>_HISTOGRAM_Gcont.svg  error histogram
    ├── error_<stem>.txt      / error_<stem>_Gcont.txt      metrics (CC / RMSE / NRMSE / rel-L2 / %err / ASPE)
    └── <stem>_*_Output*.csv                                per-point predicted and ground-truth values
```

---

## 4. Environment

Tested on:

| | |
|--|--|
| OS | Ubuntu 24.04 |
| Python | 3.13.3 |
| TensorFlow | 2.14 (GPU) |

Python packages by step:

* `1_save_normalization.py` — `numpy`, `pandas`, `scikit-learn` (no TensorFlow).
* `2_predict_and_compare.py` — `numpy`, `pandas`, `matplotlib`, and
  **TensorFlow 2.14**, the version the models were trained with (Keras 2.x `.h5`).
  Because the models are loaded with `compile=False`, the custom training
  loss/metrics and `tensorflow_addons` (AdamW) are **not** required for inference.
