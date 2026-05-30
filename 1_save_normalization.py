"""
1_save_normalization.py  --  RUN THIS ONCE
===========================================

Loads the TRAINING data **one single time** and, in that one pass, reproduces
exactly the preprocessing the original ``try1.py`` performed up to the point
where the StandardScalers are fitted -- for EVERY model family:

    n0 | gradientScalar | laplacianScalar | divergenceVector | laplacianVector

For each family it writes the per-feature normalization COEFFICIENTS
(mean / scale) to its own file:  ``normalization_weights/<output_name>_norm.pkl``
(e.g. n0OG_norm.pkl, laplacianScalarOG_norm.pkl, gradientScalarOGAll_norm.pkl, ...).

The second script (2_predict_and_compare.py) then re-uses these coefficients to
normalize the test data WITHOUT ever touching the training data again.

Why coefficients and not the sklearn objects?  We store only mean_ / scale_
numpy arrays and apply (x - mean) / scale manually later. That reproduces
StandardScaler.transform to float32 precision and is independent of any
scikit-learn version.

Dependencies: numpy, pandas, scikit-learn   (NO TensorFlow needed here.)

------------------------------------------------------------------------------
Why this reads the big CSV only once (and is still correct per-family)
------------------------------------------------------------------------------
``add_features`` is family-specific only in the ``OG{i}`` / ``ALL{i}`` columns,
which it OVERWRITES each call; every other column it touches is a deterministic
function of the raw inputs (which it never modifies). The OG/ALL row filter uses
boolean indexing, which returns a NEW frame rather than shrinking the source.

So we keep ONE master frame in memory, re-run ``add_features`` per family (which
just refreshes that family's OG/ALL columns), and slice each family's filtered
subset off the master locally -- the master's row count is never reduced, so no
family contaminates the next and no full per-family copy of the data is needed.

------------------------------------------------------------------------------
NOTE on exact reproducibility
------------------------------------------------------------------------------
The original training script shuffled the training rows with an *unseeded*
sklearn.utils.shuffle before the 80/20 train/validation split, so the precise
subset the scalers were fitted on is not perfectly recoverable. For a large
dataset the per-feature mean/std over 80% of the rows is numerically
indistinguishable from any other 80% (and from 100%), so this has no practical
effect on the normalized test inputs. A fixed seed is used below for
determinism; set FIT_ON_FULL_TRAIN = True to fit on 100% of the rows instead.
"""

import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import common_pipeline as cp


# ============================ CONFIG (match try1.py) ========================
DATA_DIR            = "data"
input_file          = f'{DATA_DIR}/dataForTrainingAndValidation.csv'
test_file           = f'{DATA_DIR}/test_near_wall.csv'

# All model families to generate coefficients for, in one pass over the data.
NAMES               = ["n0", "gradientScalar", "laplacianScalar",
                       "divergenceVector", "laplacianVector"]

OG                  = "on"                # "on" -> predict ghost contribution
psum                = True
outputNormalization = "no"                # "yes" only if you trained with output scaling

# Reproducibility knobs (see note in the docstring)
SHUFFLE_SEED        = 42
FIT_ON_FULL_TRAIN   = False               # True -> skip the 80/20 split, fit on all rows

# Folder where the .pkl coefficient files are written
NORM_DIR            = "normalization_weights"
# ===========================================================================


def fit_family(name, master, test_master, norm_dir):
    """
    Fit the scalers for a single model family off the shared master frames and
    write its <output_name>_norm.pkl. The master frames are NEVER shrunk: this
    refreshes the family-specific OG/ALL columns on them and slices the filtered
    subset into a local variable only.
    """
    # ---- feature engineering (refreshes this family's OG/ALL columns) ------
    cp.add_features(master, name)
    if test_master is not None:
        cp.add_features(test_master, name)

    # ---- OG/ALL min filter on train (try1.py lines ~565-606) --------------
    # Boolean indexing returns a NEW frame, so `master` keeps all its rows for
    # the next family.
    list_og  = [f"OG{i}"  for i in range(9)]
    list_all = [f"ALL{i}" for i in range(9)]
    train_og  = master[list_og].min(axis=1)
    train_all = master[list_all].min(axis=1)
    if test_master is not None:
        test_og_max  = test_master[list_og].min(axis=1).max()
        test_all_max = test_master[list_all].min(axis=1).max()
        if OG == "off":
            train_family = master[train_all < test_all_max]
        else:
            train_family = master[train_og < test_og_max]
    else:
        print("  WARNING: no test file -> skipping the OG/ALL train filter.")
        train_family = master

    # ---- select columns & subset ------------------------------------------
    input_columns, output_columns = cp.select_columns(name, OG)
    all_columns = input_columns + output_columns
    train_family = train_family[all_columns]
    print(f"  # input columns : {len(input_columns)}")
    print(f"  # output columns: {len(output_columns)}")
    print(f"  train rows after filter/subset: {len(train_family)}")

    # ---- reproduce the 80/20 split (scalers were fit on the 80% train) ----
    if FIT_ON_FULL_TRAIN:
        train_fit = train_family
    else:
        train_fit, _validation = train_test_split(
            train_family, test_size=0.2, random_state=42)
    print(f"  rows used to fit the scalers: {len(train_fit)}")

    # ---- fit the WIDE (flat) input scaler ---------------------------------
    wide_scaler = StandardScaler().fit(train_fit[input_columns].values)

    # ---- fit per-group CNN scalers ----------------------------------------
    group_specs = cp.cnn_group_specs(input_columns)   # ordered [(key, [9 cols]), ...]
    cnn_scalers = {}
    cnn_group_order = []
    cnn_group_columns = {}
    for key, cols in group_specs:
        sc = StandardScaler().fit(train_fit[cols].values)
        cnn_scalers[key] = {"mean": sc.mean_.astype(np.float64),
                            "scale": sc.scale_.astype(np.float64)}
        cnn_group_order.append(key)
        cnn_group_columns[key] = cols

    # ---- optional output scaler -------------------------------------------
    output_mean = output_scale = None
    if outputNormalization == "yes":
        out_sc = StandardScaler().fit(train_fit[output_columns].values)
        output_mean = out_sc.mean_.astype(np.float64)
        output_scale = out_sc.scale_.astype(np.float64)

    # ---- save --------------------------------------------------------------
    output_name = cp.get_output_name(output_columns)
    coeffs = {
        "name_family":         name,
        "OG":                  OG,
        "psum":                psum,
        "outputNormalization": outputNormalization,
        "output_name":         output_name,
        "input_columns":       input_columns,
        "output_columns":      output_columns,
        "cnn_group_order":     cnn_group_order,
        "cnn_group_columns":   cnn_group_columns,
        "cnn_scalers":         cnn_scalers,
        "wide_mean":           wide_scaler.mean_.astype(np.float64),
        "wide_scale":          wide_scaler.scale_.astype(np.float64),
        "output_mean":         output_mean,
        "output_scale":        output_scale,
    }

    out_path = norm_dir / f"{output_name}_norm.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(coeffs, f)

    print(f"  -> saved {out_path.name}  "
          f"(wide {len(input_columns)} feats, CNN groups {cnn_group_order}, "
          f"output scale {'yes' if outputNormalization == 'yes' else 'no'})")

    # free the per-family slice before the next family
    del train_family, train_fit
    gc.collect()
    return output_name


def main():
    # all paths are anchored to this script's directory, so the script can be
    # run from anywhere (e.g. `python project/1_save_normalization.py`)
    base_dir = Path(__file__).resolve().parent

    # ---- 1. load training (+ test, needed for the OG/ALL filter on train) --
    #         THIS IS THE ONLY TIME THE CSVs ARE READ.
    print("Loading training data (once) ...")
    master = cp.reduce_memory_usage(cp.import_csv(base_dir / input_file))
    print("Loading test data (once, only for the train-side OG/ALL filter) ...")
    test_master = cp.reduce_memory_usage(cp.import_csv(base_dir / test_file))
    if master is None:
        raise FileNotFoundError(f"Could not load training file: {base_dir / input_file}")

    # shuffle the training rows (as in try1.py), seeded for determinism
    master = shuffle(master, random_state=SHUFFLE_SEED)

    # ---- 2. psum value copy (idempotent; applied once to the master) -------
    if psum:
        cp.apply_psum_copy(master, test_master)

    # ---- 3. fit every family off the single master --------------------------
    norm_dir = base_dir / NORM_DIR
    norm_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFitting {len(NAMES)} families from one in-memory copy of the data:\n")
    written = []
    for name in NAMES:
        print(f"[{name}]")
        output_name = fit_family(name, master, test_master, norm_dir)
        written.append((name, output_name))
        print()

    # ---- 4. summary --------------------------------------------------------
    print("=" * 64)
    print(f"Done. Wrote {len(written)} coefficient files to {norm_dir}:")
    for name, output_name in written:
        print(f"  {name:<18} -> {output_name}_norm.pkl")


if __name__ == "__main__":
    main()