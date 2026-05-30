"""
common_pipeline.py
==================

Shared, deterministic preprocessing used by BOTH scripts:

    1) save_normalization.py    -> fits the StandardScalers on the TRAINING data
                                   once and writes the coefficients to disk.
    2) predict_and_compare.py   -> loads ONLY the test data, re-uses the saved
                                   coefficients, loads the trained network from
                                   the .h5 files, predicts and compares.

Everything in this module is taken verbatim (logic-wise) from the original
``try1.py`` so that the numbers produced by the two-script workflow are
identical to the original single-script workflow.

This module has NO TensorFlow dependency, so script (1) can run in a light
environment (pandas / numpy / scikit-learn only).
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# File loading (identical to FileImporter / reduce_memory_usage in try1.py)
# ----------------------------------------------------------------------------
def import_csv(file_name):
    """Read a CSV. ``file_name`` may be absolute or relative to the CWD."""
    path = Path(file_name)
    if not path.is_absolute():
        path = Path.cwd() / path
    print(path)
    try:
        return pd.read_csv(path)
    except Exception as e:                                    # noqa: BLE001
        print(f"Error loading file {path}: {e}")
        return None


def reduce_memory_usage(df):
    if df is None:
        return None
    if 'Unnamed: 99' in df.columns:
        df.drop(columns=['Unnamed: 99'], inplace=True)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df


# ----------------------------------------------------------------------------
# psum value-copy map (try1.py lines ~328-347)
# ----------------------------------------------------------------------------
PSUM_VALUE_COPY_MAP = {
    'gradientScalarWithSumOfScalar(X)':   'gradientScalar(X)',
    'gradientScalarWithSumOfScalar(Y)':   'gradientScalar(Y)',
    'gradientScalarWithSumOfScalarOG(X)': 'gradientScalarOG(X)',
    'gradientScalarWithSumOfScalarOG(Y)': 'gradientScalarOG(Y)',
    'gradientScalarWithSumOfScalarWG(X)': 'gradientScalarWG(X)',
    'gradientScalarWithSumOfScalarWG(Y)': 'gradientScalarWG(Y)',
}


def apply_psum_copy(*dfs):
    """Copy source -> target columns when psum == True. Operates in place."""
    for src_col, tgt_col in PSUM_VALUE_COPY_MAP.items():
        for df in dfs:
            if df is None:
                continue
            if src_col in df.columns and tgt_col in df.columns:
                df[tgt_col] = df[src_col]


# ----------------------------------------------------------------------------
# Feature engineering (try1.py add_features). ``name`` passed explicitly
# instead of relying on a global.
# ----------------------------------------------------------------------------
def add_features(data, name, num_neighbors=9):
    for i in range(num_neighbors):
        data[f"wallParticleNeighbours(dX)({i})"] = data[f"wallParticleNeighbours(X)({i})"] - data["X"]
        data[f"wallParticleNeighbours(dY)({i})"] = data[f"wallParticleNeighbours(Y)({i})"] - data["Y"]
        if name != "n0":
            data[f"wallParticleNeighboursTempratureDiff({i})"] = (
                data[f"wallParticleNeighboursTemprature({i})"] - data["temprature"])
            data[f"wallParticleNeighboursTempratureEij(X)({i})"] = (
                data[f"wallParticleNeighboursTempratureDiff({i})"] * data[f"wallParticleNeighbours(dX)({i})"])
            data[f"wallParticleNeighboursTempratureEij(Y)({i})"] = (
                data[f"wallParticleNeighboursTempratureDiff({i})"] * data[f"wallParticleNeighbours(dY)({i})"])
            data[f"wallParticleNeighboursVelocity(dX)({i})"] = (
                data[f"wallParticleNeighboursVelocity(X)({i})"] - data["velocity(X)"])
            data[f"wallParticleNeighboursVelocity(dY)({i})"] = (
                data[f"wallParticleNeighboursVelocity(Y)({i})"] - data["velocity(Y)"])
            data[f"wallParticleNeighboursVelocityEij({i})"] = (
                data[f"wallParticleNeighboursVelocity(dX)({i})"] * data[f"wallParticleNeighbours(dX)({i})"] -
                data[f"wallParticleNeighboursVelocity(dY)({i})"] * data[f"wallParticleNeighbours(dY)({i})"])
        data[f"wallParticleNeighboursDistance(pow2)({i})"] = data[f"wallParticleNeighboursDistance({i})"] ** 2
        if name == "n0":
            data[f"OG{i}"]  = data[f"wallParticleNeighboursDistance({i})"] * data["n0OG"] ** 2
            data[f"ALL{i}"] = data[f"wallParticleNeighboursDistance({i})"] * data["n0"] ** 2
        elif name == "gradientScalar":
            data[f"OG{i}"]  = data[f"wallParticleNeighboursDistance({i})"] * (
                ((data["gradientScalarOG(X)"] ** 2) + (data["gradientScalarOG(Y)"] ** 2)) ** 0.5)
            data[f"ALL{i}"] = data[f"wallParticleNeighboursDistance({i})"] * (
                ((data["gradientScalar(X)"] ** 2) + (data["gradientScalar(Y)"] ** 2)) ** 0.5)
        elif name == "laplacianScalar":
            data[f"OG{i}"]  = data[f"wallParticleNeighboursDistance({i})"] * data["laplacianScalarOG"] ** 2
            data[f"ALL{i}"] = data[f"wallParticleNeighboursDistance({i})"] * data["laplacianScalar"] ** 2
        elif name == "laplacianVector":
            data[f"OG{i}"]  = data[f"wallParticleNeighboursDistance({i})"] * (
                ((data["laplacianVectorOG(X)"] ** 2) + (data["laplacianVectorOG(Y)"] ** 2)) ** 0.5)
            data[f"ALL{i}"] = data[f"wallParticleNeighboursDistance({i})"] * (
                ((data["laplacianVector(X)"] ** 2) + (data["laplacianVector(Y)"] ** 2)) ** 0.5)
        elif name == "divergenceVector":
            data[f"OG{i}"]  = data[f"wallParticleNeighboursDistance({i})"] * data["divergenceVectorOG"] ** 2
            data[f"ALL{i}"] = data[f"wallParticleNeighboursDistance({i})"] * data["divergenceVector"] ** 2
    return data


# ----------------------------------------------------------------------------
# Column selection (try1.py select_columns)
# ----------------------------------------------------------------------------
def select_columns(name, OG):
    input_columns = [
        'X', 'Y', 'particleDiameter', 're', 'numberOfWallParticlesInNeghbs', 'lamda',
        *[f'wallParticleNeighbours(X)({i})' for i in range(9)],
        *[f'wallParticleNeighbours(Y)({i})' for i in range(9)],
        *[f'wallParticleNeighbours(dX)({i})' for i in range(9)],
        *[f'wallParticleNeighbours(dY)({i})' for i in range(9)],
        *[f'wallParticleNeighboursDistance(pow2)({i})' for i in range(9)],
        *[f'wallParticleNeighboursDistance({i})' for i in range(9)],
    ]
    output_columns = []

    if name == "n0":
        input_columns += ["n0WG"]
        output_columns = ['n0OG'] if OG == "on" else ['n0']

    elif name == "gradientScalar":
        input_columns += [
            'n0', 'temprature', 'gradientScalarWG(X)', 'gradientScalarWG(Y)',
            *[f'wallParticleNeighboursBCTypes({i})' for i in range(9)],
            *[f'wallParticleNeighboursTemprature({i})' for i in range(9)],
            *[f'wallParticleNeighboursTempratureDiff({i})' for i in range(9)],
            *[f'wallParticleNeighboursTempratureEij(X)({i})' for i in range(9)],
            *[f'wallParticleNeighboursTempratureEij(Y)({i})' for i in range(9)],
        ]
        output_columns = (['gradientScalarOG(X)', 'gradientScalarOG(Y)'] if OG == "on"
                          else ['gradientScalar(X)', 'gradientScalar(Y)'])

    elif name == "laplacianScalar":
        input_columns += [
            'n0', 'temprature', 'laplacianScalarWG',
            *[f'wallParticleNeighboursBCTypes({i})' for i in range(9)],
            *[f'wallParticleNeighboursTemprature({i})' for i in range(9)],
            *[f'wallParticleNeighboursTempratureDiff({i})' for i in range(9)],
            *[f'wallParticleNeighboursTempratureEij(X)({i})' for i in range(9)],
            *[f'wallParticleNeighboursTempratureEij(Y)({i})' for i in range(9)],
        ]
        output_columns = ['laplacianScalarOG'] if OG == "on" else ['laplacianScalar']

    elif name == "divergenceVector":
        input_columns += [
            'n0', 'velocity(X)', 'velocity(Y)', 'divergenceVectorWG',
            *[f'wallParticleNeighboursBCTypes({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(X)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(Y)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(dX)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(dY)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocityEij({i})' for i in range(9)],
        ]
        output_columns = ['divergenceVectorOG'] if OG == "on" else ['divergenceVector']

    elif name == "laplacianVector":
        input_columns += [
            'n0', 'velocity(X)', 'velocity(Y)', 'laplacianVectorWG(X)', 'laplacianVectorWG(Y)',
            *[f'wallParticleNeighboursBCTypes({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(X)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(Y)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(dX)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocity(dY)({i})' for i in range(9)],
            *[f'wallParticleNeighboursVelocityEij({i})' for i in range(9)],
        ]
        output_columns = (['laplacianVectorOG(X)', 'laplacianVectorOG(Y)'] if OG == "on"
                          else ['laplacianVector(X)', 'laplacianVector(Y)'])

    return input_columns, output_columns


# ----------------------------------------------------------------------------
# CNN stencil groups, in the EXACT order makingNN builds them. This single
# ordered list drives both (a) per-group normalization and (b) the order the
# model inputs are fed in. ``key`` is just a label; ``base`` builds the 9
# neighbour columns "base(0)" .. "base(8)".
# ----------------------------------------------------------------------------
BASE_CNN_GROUPS = [
    ("x",             "wallParticleNeighbours(X)"),
    ("y",             "wallParticleNeighbours(Y)"),
    ("dx",            "wallParticleNeighbours(dX)"),
    ("dy",            "wallParticleNeighbours(dY)"),
    ("distance",      "wallParticleNeighboursDistance"),
    ("distance_pow2", "wallParticleNeighboursDistance(pow2)"),
]
TEMP_CNN_GROUPS = [
    ("BCTypes",           "wallParticleNeighboursBCTypes"),
    ("temperature",       "wallParticleNeighboursTemprature"),
    ("temperature_diff",  "wallParticleNeighboursTempratureDiff"),
    ("temperature_eij_x", "wallParticleNeighboursTempratureEij(X)"),
    ("temperature_eij_y", "wallParticleNeighboursTempratureEij(Y)"),
]
VEL_CNN_GROUPS = [
    ("BCTypes",      "wallParticleNeighboursBCTypes"),
    ("velocity_x",   "wallParticleNeighboursVelocity(X)"),
    ("velocity_y",   "wallParticleNeighboursVelocity(Y)"),
    ("velocity_dx",  "wallParticleNeighboursVelocity(dX)"),
    ("velocity_dy",  "wallParticleNeighboursVelocity(dY)"),
    ("velocity_eij", "wallParticleNeighboursVelocityEij"),
]


def cnn_group_specs(input_columns):
    """Return an ordered list of (group_key, [9 column names]) matching makingNN."""
    groups = list(BASE_CNN_GROUPS)
    if "temprature" in input_columns:
        groups += TEMP_CNN_GROUPS
    if "velocity(X)" in input_columns:
        groups += VEL_CNN_GROUPS
    return [(key, [f"{base}({i})" for i in range(9)]) for key, base in groups]


# ----------------------------------------------------------------------------
# Output-file naming (try1.py output_column_map / get_output_name).
# This decides which {name}.h5 / {name}_weights.h5 to load.
# ----------------------------------------------------------------------------
OUTPUT_COLUMN_MAP = {
    'n0OG': 'n0OG',
    'lamdaOG': 'lamdaOG',
    'laplacianScalarOG': 'laplacianScalarOG',
    'divergenceVectorOG': 'divergenceVectorOG',
    ('laplacianVectorOG(X)', 'laplacianVectorOG(Y)'): 'laplacianVectorOGAll',
    'laplacianVectorOG(X)': 'laplacianVectorOG(X)',
    'laplacianVectorOG(Y)': 'nlaplacianVectorOG(Y)',
    ('gradientScalarOG(X)', 'gradientScalarOG(Y)'): 'gradientScalarOGAll',
    'gradientScalarOG(X)': 'gradientScalarOG(X)',
    'gradientScalarOG(Y)': 'gradientSc2alarOG(Y)',
    'n0': 'n0',
    'lamda': 'lamda',
    'laplacianScalar': 'laplacianScalar',
    'divergenceVector': 'divergenceVector',
    ('laplacianVector(X)', 'laplacianVector(Y)'): 'laplacianVectorAll',
    'laplacianVector(X)': 'laplacianVector(X)',
    'laplacianVector(Y)': 'nlaplacianVector(Y)',
    ('gradientScalar(X)', 'gradientScalar(Y)'): 'gradientScalarAll',
    'gradientScalar(X)': 'gradientScalar(X)',
    'gradientScalar(Y)': 'gradientScalar(Y)',
}


def get_output_name(output_columns):
    for keys, value in OUTPUT_COLUMN_MAP.items():
        if isinstance(keys, tuple):
            if all(k in output_columns for k in keys):
                return value
        elif keys in output_columns:
            return value
    return None


# ----------------------------------------------------------------------------
# WG ("with-ghost" baseline) columns used to reconstruct the TOTAL field from
# the OG (ghost-contribution-only) prediction. Mirrors process_output_columns
# + output_adjustments in try1.py, but only what the test set needs.
# Returns a dict matching the weight-name keys used by adjust_output.
# ----------------------------------------------------------------------------
def extract_wg_for_output(data, output_columns):
    """Return WG values needed to add back to the OG prediction/ground-truth."""
    wg = {}
    if 'n0OG' in output_columns:
        wg['n0WG'] = data['n0WG'].values
    if 'laplacianScalarOG' in output_columns:
        wg['laplacianScalarWG'] = data['laplacianScalarWG'].values
    if 'divergenceVectorOG' in output_columns:
        wg['divergenceVectorWG'] = data['divergenceVectorWG'].values
    if 'laplacianVectorOG(X)' in output_columns and 'laplacianVectorOG(Y)' in output_columns:
        wg['laplacianVectorWGX'] = data['laplacianVectorWG(X)'].values
        wg['laplacianVectorWGY'] = data['laplacianVectorWG(Y)'].values
    if 'gradientScalarOG(X)' in output_columns and 'gradientScalarOG(Y)' in output_columns:
        wg['gradientScalarWGX'] = data['gradientScalarWG(X)'].values
        wg['gradientScalarWGY'] = data['gradientScalarWG(Y)'].values
    return wg


# Maps an output_columns membership test to the WG key(s) (try1.py
# output_adjustments). Used to add WG back for the TOTAL ("All") round.
OUTPUT_ADJUSTMENTS = {
    'n0OG':               'n0WG',
    'laplacianScalarOG':  'laplacianScalarWG',
    'divergenceVectorOG': 'divergenceVectorWG',
    ('laplacianVectorOG(X)', 'laplacianVectorOG(Y)'): ('laplacianVectorWGX', 'laplacianVectorWGY'),
    ('gradientScalarOG(X)', 'gradientScalarOG(Y)'):   ('gradientScalarWGX', 'gradientScalarWGY'),
}


def add_wg_back(arr, output_columns, wg):
    """Add the WG baseline back to a (rows, n_out) array. Returns a new array."""
    out = np.array(arr, dtype=float, copy=True)
    for columns, weight_name in OUTPUT_ADJUSTMENTS.items():
        multi_dim = isinstance(columns, tuple)
        if multi_dim:
            if all(c in output_columns for c in columns):
                out[:, 0] += wg[weight_name[0]]
                out[:, 1] += wg[weight_name[1]]
        elif columns in output_columns:
            out[:, 0] += wg[weight_name]
    return out


# ----------------------------------------------------------------------------
# Metrics (numpy re-implementation of RMSEAndCC in try1.py; same formulas).
# ----------------------------------------------------------------------------
def _cc(a, b):
    am, bm = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.sum(am * am) * np.sum(bm * bm)) + 1e-7
    return float(np.sum(am * bm) / denom)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _nrmse_pct(a, b):
    return float((np.sqrt(np.mean((a - b) ** 2)) / (a.max() - a.min())) * 100.0)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8))


def _percentage_error(y, y_pred):
    eps1 = abs(min(y.min(), y_pred.min()))
    eps2 = abs(max(y.max(), y_pred.max()))
    epsilon = 0.05 * max(eps1, eps2)
    y_safe = np.where(np.abs(y) < epsilon, epsilon, y)
    y_pred_safe = np.where(np.abs(y_pred) < epsilon, epsilon, y_pred)
    return float(np.mean(np.abs((y_safe - y_pred_safe) / y_safe)) * 100.0)


def _aspe(y, y_pred, epsilon=1e-8):
    norm_factor = (y.max() - y.min()) + epsilon
    error = np.clip(np.abs(y - y_pred) / norm_factor, 0.0, 1.0)
    return float(np.mean(error) * 100.0)


def rmse_and_cc(A, B, label):
    """numpy port of RMSEAndCC: drops indices where A==0, flattens, prints."""
    A = np.asarray(A, dtype=np.float64).reshape(-1)
    B = np.asarray(B, dtype=np.float64).reshape(-1)
    zero_idx = np.where(A == 0)[0]
    A = np.delete(A, zero_idx)
    B = np.delete(B, zero_idx)

    cc = _cc(A, B)
    rmse = _rmse(A, B)
    nrmse = _nrmse_pct(A, B)
    rel_l2 = _rel_l2(A, B)
    pct = _percentage_error(A, B)
    aspe = _aspe(A, B)

    print(f"{label} - CC: {cc:.4f}")
    print(f"{label} - RMSE: {rmse:.4f}")
    print(f"{label} - NRMSE: {nrmse:.4f}%")
    print(f"{label} - Relative L2 Error: {rel_l2:.4f}")
    print(f"{label} - Percentage Error: {pct:.2f}%")
    print(f"{label} - ASPE: {aspe:.2f}%")
    return cc, rmse, nrmse, rel_l2, pct, aspe


def evaluate_test_only(true_data, predicted_data, output_file_name):
    """
    Evaluate only the 'Test (unseen)' pair and write the error file.
    Produces the keys plot_configs expects: cc_Test (unseen), rmse_Test
    (unseen) and the _x / _y variants for vector outputs.
    """
    true_data = np.asarray(true_data)
    predicted_data = np.asarray(predicted_data)
    results = {}
    ds = "Test (unseen)"
    if true_data.shape[1] == 1:
        cc, rmse, nrmse, rel_l2, pct, aspe = rmse_and_cc(true_data, predicted_data, f"{ds}_(Overall)")
        results.update({f"cc_{ds}": cc, f"rmse_{ds}": rmse, f"nrmse_{ds}%": nrmse,
                        f"aspe_{ds}": aspe, f"relL2_{ds}": rel_l2, f"percentileError_{ds}": pct})
    else:
        cx = rmse_and_cc(true_data[:, 0], predicted_data[:, 0], f"{ds} (X)")
        cy = rmse_and_cc(true_data[:, 1], predicted_data[:, 1], f"{ds} (Y)")
        co = rmse_and_cc(true_data, predicted_data, f"{ds}_(Overall)")
        results.update({
            f"cc_{ds}": co[0], f"rmse_{ds}": co[1], f"nrmse_{ds}%": co[2],
            f"aspe_{ds}": co[5], f"relL2_{ds}": co[3], f"percentileError_{ds}": co[4],
            f"cc_{ds}_x": cx[0], f"rmse_{ds}_x": cx[1], f"nrmse_{ds}_x%": cx[2],
            f"aspe_{ds}_x": cx[5], f"relL2_{ds}_x": cx[3], f"percentileError_{ds}_x": cx[4],
            f"cc_{ds}_y": cy[0], f"rmse_{ds}_y": cy[1], f"nrmse_{ds}_y%": cy[2],
            f"aspe_{ds}_y": cy[5], f"relL2_{ds}_y": cy[3], f"percentileError_{ds}_y": cy[4],
        })

    with open(output_file_name, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {round(float(value), 3)}\n")
    print(f"Results saved to {output_file_name}")
    return results
