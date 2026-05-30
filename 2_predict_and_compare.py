"""
predict_and_compare.py  --  INFERENCE ONLY
===========================================

Does NOT load the training data and does NOT rebuild/train the network.

It:
  1. loads ONLY the test files,
  2. loads the normalization coefficients written by save_normalization.py,
  3. loads the trained network straight from the .h5 files
        - structure  from  "<name>.h5"          (load_model, compile=False)
        - weights    from  "<name>_weights.h5"   (load_weights)
  4. normalizes the test inputs with the saved coefficients,
  5. predicts,
  6. evaluates (CC / RMSE / NRMSE / rel-L2 / %err / ASPE) and
     plots test vs. predicted (spatial maps, predicted-vs-ground-truth
     scatter, error histogram), exactly like the original try1.py.

When OG == "on" two rounds are produced (same as the original):
  * "_Gcont"  -> the raw ghost-contribution prediction
  *  total    -> after adding the WG baseline back (the full reconstructed field)

Dependencies: numpy, pandas, matplotlib, tensorflow (== the version used for
training, Keras 2.x).  scikit-learn is NOT required here.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

import common_pipeline as cp

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
tf.get_logger().setLevel('ERROR')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 15})


# ============================ CONFIG (match try1.py) ========================
DATA_DIR             = "data"
test_file            = f'{DATA_DIR}/test_near_wall.csv'
test_file_all_points = f'{DATA_DIR}/test_not_near_wall.csv'
# ["n0", "gradientScalar", "laplacianScalar", "divergenceVector", "laplacianVector"]
name                = "divergenceVector"   # MUST match what you used in 1_save_normalization.py
OG                  = "on"
psum                = True
outputNormalization = "no"

saveFig             = "on"
batch_size          = int(16384)
verbose             = 2

# Where the coefficients + model live. Leave MODEL_BASENAME = None to derive it
# from the output columns (get_output_name). Override it for renamed files,
# e.g. "gradientScalarOGAllMJ" or the OG-off "divergenceVector".
NORM_FILE      = None          # None -> "<output_name>_norm.pkl"
MODEL_BASENAME = None          # None -> get_output_name(output_columns)

# Folders
NN_MODEL_DIR = "model"                  # contains <name>.h5 and <name>_weights.h5
NORM_DIR     = "normalization_weights"  # contains <name>_norm.pkl
# Comparison outputs (figures / error_*.txt / *.csv) go into a per-operator
# subfolder under OUTPUT_ROOT:  output/<name>/  (e.g. output/laplacianScalar/,
# output/n0/, output/gradientScalar/, ...). Leave OUTPUT_DIR = None to use
# ``name`` for the subfolder; set it to override just the subfolder name.
OUTPUT_ROOT  = "output"
OUTPUT_DIR   = None
# ===========================================================================


def calculate_difference(output_test, output_eval, plot_name):
    """try1.py calculate_difference (n0 normalised by ground truth)."""
    if plot_name in ["n0", "n0OG"]:
        output_eval = np.divide(output_eval, output_test)
        output_test = np.divide(output_test, output_test)
        difference = np.divide(np.subtract(output_eval, output_test), output_test)
    else:
        difference = np.subtract(output_eval, output_test)
    return output_test, output_eval, difference


def plot_data(output_test_data, output_evaluated, plot_name, save_figure,
              test_points_allPoints, test_points, suffix, out_dir):
    """Faithful port of try1.py plotData. ``suffix`` is "_Gcont" or ""."""
    output_test_data, output_evaluated, difference = calculate_difference(
        output_test_data, output_evaluated, plot_name)
    cmap = mpl.colormaps.get_cmap('seismic')

    x_margin = test_points["particleDiameter"].iloc[0] * 2
    x_min, x_max = test_points["X"].min() - x_margin, test_points["X"].max() + x_margin
    y_min, y_max = test_points["Y"].min() - x_margin, test_points["Y"].max() + x_margin

    if output_test_data.shape[1] == 1:
        # ----- spatial: ground truth vs predicted -----
        fig, axes = plt.subplots(1, 2, sharey=True, constrained_layout=True)
        fig.set_size_inches(12, 6)
        if plot_name in ["n0", "n0OG"]:
            vall1 = np.ones_like(test_points_allPoints["X"])
        else:
            vall1 = np.zeros_like(test_points_allPoints["X"])
        vmin = min(np.min(output_test_data), np.min(output_evaluated))
        vmax = max(np.max(output_test_data), np.max(output_evaluated))

        axes[0].scatter(test_points_allPoints["X"], test_points_allPoints["Y"], c=vall1,
                        vmin=vmin, vmax=vmax, cmap=cmap, s=6)
        axes[0].scatter(test_points["X"], test_points["Y"], c=output_test_data[:, 0],
                        vmin=vmin, vmax=vmax, cmap=cmap, s=6)
        axes[0].axis("equal"); axes[0].set_title("Ground Truth")
        axes[0].set_xlim([x_min, x_max]); axes[0].set_ylim([y_min, y_max])

        axes[1].scatter(test_points_allPoints["X"], test_points_allPoints["Y"], c=vall1,
                        vmin=vmin, vmax=vmax, cmap=cmap, s=6)
        ax1 = axes[1].scatter(test_points["X"], test_points["Y"], c=output_evaluated[:, 0],
                              vmin=vmin, vmax=vmax, cmap=cmap, s=6)
        axes[1].axis("equal"); axes[1].set_title("Predicted")
        axes[1].set_xlim([x_min, x_max]); axes[1].set_ylim([y_min, y_max])
        plt.setp(axes, xlabel="X (m)"); axes[0].set_ylabel("Y (m)")
        fig.colorbar(ax1, ax=axes.ravel().tolist(), location='bottom', shrink=0.6)
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}{suffix}.svg")

        # ----- difference map -----
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        dmax = np.max(np.abs(difference))
        ax1 = ax.scatter(test_points["X"], test_points["Y"], c=difference[:, 0],
                         vmin=-dmax, vmax=dmax, cmap=cmap, s=6)
        ax.axis("equal"); ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max])
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        fig.colorbar(ax1, ax=ax, location='bottom', shrink=0.8)
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}_diff{suffix}.svg")

        # ----- predicted vs ground truth scatter -----
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(6, 6)
        ax.plot([-100000, 100000], [-100000, 100000], color='blue')
        ax.scatter(output_evaluated, output_test_data, color='red', s=5)
        ax.axis("equal")
        if plot_name in ["n0", "n0OG"]:
            ax.set_xlim([0.5, 1.5]); ax.set_ylim([0.5, 1.5])
        else:
            lo = min(np.min(output_test_data), np.min(output_evaluated))
            hi = max(np.max(output_test_data), np.max(output_evaluated))
            ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}_PVSGT{suffix}.svg")

        # ----- error histogram -----
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(6, 6)
        ax.hist(difference[:, 0], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='dashed', linewidth=1)
        ax.set_xlabel("Prediction Error"); ax.set_ylabel("Frequency"); ax.grid(True)
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}_HISTOGRAM{suffix}.svg")

    else:
        # ============ vector output (X & Y components) ============
        vmin = min(np.min(output_test_data[:, :]), np.min(output_evaluated[:, :]))
        vmax = max(np.max(output_test_data[:, :]), np.max(output_evaluated[:, :]))

        fig, axes = plt.subplots(2, 2, sharey=True, constrained_layout=True)
        fig.set_size_inches(12, 12)
        vallplot = np.zeros_like(test_points_allPoints["X"])
        for idx, dir_name in enumerate(['(X-dir.)', '(Y-dir.)']):
            axes[idx, 0].scatter(test_points_allPoints["X"], test_points_allPoints["Y"],
                                 c=vallplot, vmin=vmin, vmax=vmax, cmap=cmap, s=6)
            axes[idx, 0].scatter(test_points["X"], test_points["Y"], c=output_test_data[:, idx],
                                 vmin=vmin, vmax=vmax, cmap=cmap, s=6)
            axes[idx, 0].axis("equal"); axes[idx, 0].set_title(f"Ground Truth {dir_name}")
            axes[idx, 0].set_xlim([x_min, x_max]); axes[idx, 0].set_ylim([y_min, y_max])

            axes[idx, 1].scatter(test_points_allPoints["X"], test_points_allPoints["Y"],
                                 c=vallplot, vmin=vmin, vmax=vmax, cmap=cmap, s=6)
            ax1 = axes[idx, 1].scatter(test_points["X"], test_points["Y"], c=output_evaluated[:, idx],
                                       vmin=vmin, vmax=vmax, cmap=cmap, s=6)
            axes[idx, 1].axis("equal"); axes[idx, 1].set_title(f"Predicted {dir_name}")
            axes[idx, 1].set_xlim([x_min, x_max]); axes[idx, 1].set_ylim([y_min, y_max])
        plt.setp(axes[1, :], xlabel="X (m)"); plt.setp(axes[:, 0], ylabel="Y (m)")
        fig.colorbar(ax1, ax=axes.ravel().tolist(), location='bottom', shrink=0.6)
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}{suffix}.svg")

        # ----- difference maps -----
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        fig.set_size_inches(6, 12)
        dmax = np.max(np.abs(difference[:, :]))
        vall = np.zeros_like(test_points_allPoints["X"])
        for idx, dir_name in enumerate(['(X-dir.)', '(Y-dir.)']):
            ax[idx].scatter(test_points_allPoints["X"], test_points_allPoints["Y"], c=vall,
                            vmin=-dmax, vmax=dmax, cmap=cmap, s=6)
            ax1 = ax[idx].scatter(test_points["X"], test_points["Y"], c=difference[:, idx],
                                  vmin=-dmax, vmax=dmax, cmap=cmap, s=6)
            ax[idx].axis("equal"); ax[idx].set_xlim([x_min, x_max]); ax[idx].set_ylim([y_min, y_max])
            ax[idx].set_title(f"{dir_name}"); ax[idx].set_ylabel("Y (m)")
            if idx == 1:
                ax[idx].set_xlabel("X (m)")
        fig.colorbar(ax1, ax=ax.ravel().tolist(), location='bottom', shrink=0.6)
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}_diff{suffix}.svg")

        # ----- predicted vs ground truth -----
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        fig.set_size_inches(6, 12)
        for idx, dir_name in enumerate(['(X-dir.)', '(Y-dir.)']):
            ax[idx].plot([-100000, 100000], [-100000, 100000], color='blue')
            ax[idx].scatter(output_evaluated[:, idx], output_test_data[:, idx], color='red', s=6)
            ax[idx].axis("equal")
            lo = min(np.min(output_test_data[:, idx]), np.min(output_evaluated[:, idx]))
            hi = max(np.max(output_test_data[:, idx]), np.max(output_evaluated[:, idx]))
            ax[idx].set_xlim([lo, hi]); ax[idx].set_ylim([lo, hi])
            ax[idx].set_ylabel("Ground Truth"); ax[idx].set_title(f"{dir_name}")
            if idx == 1:
                ax[idx].set_xlabel("Predicted")
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}_PVSGT{suffix}.svg")

        # ----- error histograms -----
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        fig.set_size_inches(6, 12)
        for idx, dir_name in enumerate(['(X-dir.)', '(Y-dir.)']):
            ax[idx].hist(difference[:, idx], bins=50, edgecolor='black', alpha=0.7)
            ax[idx].axvline(x=0, color='red', linestyle='dashed', linewidth=1)
            ax[idx].set_ylabel("Frequency"); ax[idx].set_title(dir_name); ax[idx].grid(True)
            if idx == 1:
                ax[idx].set_xlabel("Prediction Error")
        if save_figure == "on":
            plt.savefig(out_dir / f"{plot_name}_HISTOGRAM{suffix}.svg")

    plt.close('all')


def build_plot_configs(output_columns):
    """Which field(s) to plot. Names map to the plotData filename stem."""
    cfgs = {
        'n0OG': 'n0OG', 'laplacianScalarOG': 'laplacianScalarOG',
        'divergenceVectorOG': 'divergenceVectorOG',
        'laplacianVectorOG(X)': 'laplacianVectorOG(X)', 'laplacianVectorOG(Y)': 'laplacianVectorOG(Y)',
        'gradientScalarOG(X)': 'gradientScalarOG(X)', 'gradientScalarOG(Y)': 'gradientScalarOG(Y)',
        'n0': 'n0', 'laplacianScalar': 'laplacianScalar', 'divergenceVector': 'divergenceVector',
        'gradientScalar(X)': 'gradientScalar(X)', 'gradientScalar(Y)': 'gradientScalar(Y)',
        'laplacianVector(X)': 'laplacianVector(X)', 'laplacianVector(Y)': 'laplacianVector(Y)',
    }
    return {k: v for k, v in cfgs.items() if k in output_columns}


VECTOR_KEYS = {'gradientScalarOG(X)', 'gradientScalarOG(Y)', 'laplacianVectorOG(X)', 'laplacianVectorOG(Y)',
               'gradientScalar(X)', 'gradientScalar(Y)', 'laplacianVector(X)', 'laplacianVector(Y)'}


def run_plots(evaluated_df, test_df, output_columns, test_data_all_points, test_data, suffix, out_dir):
    """Iterate plot_configs and call plot_data, like the loop in try1.py."""
    plot_configs = build_plot_configs(output_columns)
    done_vector = False
    for key, plot_name in plot_configs.items():
        if key in VECTOR_KEYS:
            base = key.replace("(X)", "").replace("(Y)", "")
            x_key, y_key = f"{base}(X)", f"{base}(Y)"
            if x_key in output_columns and y_key in output_columns:
                if done_vector:
                    continue
                output_evaluated = evaluated_df[[x_key, y_key]].values
                output_test_data = test_df[[x_key, y_key]].values
                plot_name = base            # e.g. gradientScalarOG
                done_vector = True
            else:
                continue
        else:
            output_evaluated = evaluated_df[[key]].values
            output_test_data = test_df[[key]].values
        plot_data(output_test_data, output_evaluated, plot_name, saveFig,
                  test_data_all_points, test_data, suffix, out_dir)


def main():
    # all paths are anchored to this script's directory, so the script can be
    # run from anywhere (e.g. `python project/2_predict_and_compare.py`)
    base_dir = Path(__file__).resolve().parent

    # ---- 1. load saved normalization coefficients --------------------------
    input_columns, output_columns = cp.select_columns(name, OG)
    output_name = cp.get_output_name(output_columns)
    norm_dir = base_dir / NORM_DIR
    norm_path = norm_dir / (NORM_FILE if NORM_FILE else f"{output_name}_norm.pkl")
    with open(norm_path, "rb") as f:
        coeffs = pickle.load(f)
    print("Loaded normalization coefficients from:", norm_path)

    # comparison outputs go into output/<operator>/ (e.g. output/laplacianScalar)
    out_dir = base_dir / OUTPUT_ROOT / (OUTPUT_DIR if OUTPUT_DIR else name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity check that the coefficients match this configuration
    if coeffs["input_columns"] != input_columns or coeffs["output_columns"] != output_columns:
        raise ValueError("Saved coefficients do not match the current name/OG. "
                         "Re-run save_normalization.py with the same settings.")

    # ---- 2. load TEST data only -------------------------------------------
    print("Loading test data ...")
    test_data = cp.reduce_memory_usage(cp.import_csv(base_dir / test_file))
    test_data_all_points = cp.reduce_memory_usage(cp.import_csv(base_dir / test_file_all_points))
    if test_data is None:
        raise FileNotFoundError(f"Could not load test file: {base_dir / test_file}")

    if psum:
        cp.apply_psum_copy(test_data, test_data_all_points)

    test_data = cp.add_features(test_data, name)

    # X, Y and particleDiameter are already part of input_columns, so all_columns
    # keeps everything plot_data needs (coordinates + WG baseline + ground truth).
    all_columns = input_columns + output_columns
    test_data = test_data[all_columns]
    print(f"test rows: {len(test_data)}")

    # ---- 3. ground truth (OG) ---------------------------------------------
    test_output = test_data[output_columns].values.astype(np.float32)

    # ---- 4. normalize the inputs with the SAVED coefficients --------------
    wide_input = ((test_data[input_columns].values - coeffs["wide_mean"])
                  / coeffs["wide_scale"]).astype(np.float32)

    cnn_inputs = []
    for key in coeffs["cnn_group_order"]:
        cols = coeffs["cnn_group_columns"][key]
        sc = coeffs["cnn_scalers"][key]
        g = (test_data[cols].values - sc["mean"]) / sc["scale"]
        cnn_inputs.append(g.reshape(-1, 9, 1).astype(np.float32))

    model_inputs = cnn_inputs + [wide_input]   # wide input is LAST (matches makingNN)

    # ---- 5. load the network: structure + weights from the .h5 files ------
    basename = MODEL_BASENAME if MODEL_BASENAME else output_name
    nn_dir = base_dir / NN_MODEL_DIR
    structure_h5 = nn_dir / f"{basename}.h5"
    weights_h5   = nn_dir / f"{basename}_weights.h5"
    print(f"Loading model structure from: {structure_h5}")
    model = load_model(structure_h5, compile=False)   # compile=False -> no custom loss/metrics needed
    print(f"Loading model weights   from: {weights_h5}")
    model.load_weights(weights_h5)
    model.summary()

    if len(model.inputs) != len(model_inputs):
        raise ValueError(f"Model expects {len(model.inputs)} inputs but {len(model_inputs)} were assembled. "
                         "Check that name/OG match the loaded model.")

    # ---- 6. predict --------------------------------------------------------
    evaluated_Output = model.predict(model_inputs, batch_size=batch_size, verbose=verbose)

    # ---- 7. optional output de-normalization ------------------------------
    if outputNormalization == "yes":
        o_mean, o_scale = coeffs["output_mean"], coeffs["output_scale"]
        evaluated_Output = evaluated_Output * o_scale + o_mean
        test_output = test_output * o_scale + o_mean

    # =========================================================================
    # ROUND 1 - ghost-contribution only ("_Gcont"), the raw network output
    # =========================================================================
    print("\n===== Round 1: ghost contribution (raw OG prediction) =====")
    cp.evaluate_test_only(test_output, evaluated_Output,
                          out_dir / f"error_{output_name}_Gcont.txt")

    evaluated_df = pd.DataFrame(evaluated_Output, columns=output_columns)
    test_df = pd.DataFrame(test_output, columns=output_columns)

    # save the per-point CSVs (like try1.py)
    evaluated_df.to_csv(out_dir / f"{output_name}_evaluated_Output_Gcont.csv", index=False)
    test_df.to_csv(out_dir / f"{output_name}_test_output_Gcont.csv", index=False)

    if saveFig == "on":
        run_plots(evaluated_df, test_df, output_columns,
                  test_data_all_points, test_data, "_Gcont", out_dir)

    # =========================================================================
    # ROUND 2 - total field = OG prediction + WG baseline (only when OG == on)
    # =========================================================================
    if OG == "on":
        print("\n===== Round 2: total field (OG + WG) =====")
        wg = cp.extract_wg_for_output(test_data, output_columns)
        evaluated_total = cp.add_wg_back(evaluated_Output, output_columns, wg)
        test_total = cp.add_wg_back(test_output, output_columns, wg)

        cp.evaluate_test_only(test_total, evaluated_total,
                              out_dir / f"error_{output_name}.txt")

        evaluated_total_df = pd.DataFrame(evaluated_total, columns=output_columns)
        test_total_df = pd.DataFrame(test_total, columns=output_columns)
        evaluated_total_df.to_csv(out_dir / f"{output_name}_evaluated_Output.csv", index=False)
        test_total_df.to_csv(out_dir / f"{output_name}_test_output.csv", index=False)

        if saveFig == "on":
            run_plots(evaluated_total_df, test_total_df, output_columns,
                      test_data_all_points, test_data, "", out_dir)

    print("\nDone. Figures (.svg), error_*.txt and *_Output*.csv written to:", out_dir)


if __name__ == "__main__":
    main()