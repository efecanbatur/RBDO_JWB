import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import kriging_model

casePath = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(casePath, "rbdo_dataset.csv")

INPUT_COLUMNS = [
    "M", "alpha_deg",
    "W_root_chord_0", "W_root_chord_1", "W_root_chord_2",
    "W_tip_chord_0", "W_tip_chord_1", "W_tip_chord_2",
    "W_span_0", "W_span_1", "W_span_2",
    "W_sweep_0", "W_sweep_1", "W_sweep_2"
]

OUTPUT_COLUMNS = ["CL", "CD", "PLdB"]

MODEL_FILENAME_TEMPLATE = "rbdo_model_{output_name}.joblib"

REPORT_PDF = os.path.join(casePath, "rbdo_kriging_surrogate_report.pdf")

TEST_FRACTION = 0.2
RANDOM_SEED = 42


def split_train_test_indices(n_total, test_frac=0.2, seed=42):
    if n_total < 5: 
        print("WARNING: Very few data points, using all for training.")
        return np.arange(n_total), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_test = max(1, int(test_frac * n_total))

    n_test = min(n_test, n_total - 1) if n_total > 1 else n_test

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx

def calculate_metrics(y_true, y_pred):

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan} # Default to NaN
    if len(y_true) == 0:
        print("Warning: No valid finite pairs for metrics calculation.")
        return metrics # Return NaNs if no valid data

    metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred)**2))
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred)**2)

    mean_y_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y_true)**2)

    if ss_tot > 1e-12: 
        metrics['r2'] = 1.0 - ss_res / ss_tot
    elif ss_res < 1e-12: 
        metrics['r2'] = 1.0
    else: 
        metrics['r2'] = 0.0

    return metrics

def main():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}\n"
                             f"Please run 'doe.py' first to generate the dataset.")
    df = pd.read_csv(csv_path)

    df_clean = df.dropna(subset=INPUT_COLUMNS + OUTPUT_COLUMNS).copy()
    if len(df_clean) < len(df):
        print(f"Dropped {len(df) - len(df_clean)} rows due to NaN values.")
        if len(df_clean) < 10:
             print("WARNING: Less than 10 valid data points remain. Model quality may be poor.")
    if len(df_clean) == 0:
        raise ValueError("No valid data rows remain after dropping NaNs. Cannot build models.")

    required_cols_check = set(INPUT_COLUMNS + OUTPUT_COLUMNS)
    if not required_cols_check.issubset(df_clean.columns):
        missing_cols = required_cols_check - set(df_clean.columns)
        raise ValueError(f"Cleaned CSV is missing required columns: {missing_cols}")

    X_data_full = df_clean[INPUT_COLUMNS].to_numpy(dtype=float)
    n_total, n_dims = X_data_full.shape
    print(f"Loaded {n_total} clean samples with {n_dims} input dimensions.")
    print(f"Input dimensions are: {INPUT_COLUMNS}")

    with PdfPages(REPORT_PDF) as pdf:

        fig_summary, ax_summary = plt.subplots(figsize=(8.5, 11))
        ax_summary.axis("off")
        ax_summary.text(0.05, 0.95, "RBDO Kriging Surrogate Audit Result", fontsize=16, weight='bold')
        summary_text = (
            f"Data Source: {os.path.basename(csv_path)}\n"
            f"Total Clean Samples (N): {n_total}\n"
            f"Input Dimensions ({n_dims}): {', '.join(INPUT_COLUMNS)}\n"
            f"Outputs Modeled: {', '.join(OUTPUT_COLUMNS)}\n"
            f"Train/Test Split Ratio: {1-TEST_FRACTION:.0%}/{TEST_FRACTION:.0%}\n"
            f"Random Seed: {RANDOM_SEED}\n\n"
            f"Method: Kriging (Gaussian Process Regressor via sklearn)\n"
            f"Kernel: Default in kriging_model.py (likely C*RBF + WhiteKernel)\n"
            f"Note: Models fitted using standardized inputs/outputs.\n\n"
        )
        current_y_pos = 0.75

        for output_name in OUTPUT_COLUMNS:
            print(f"Processing model for: {output_name} ---")

            y_data_full = df_clean[output_name].to_numpy(dtype=float)
            valid_indices = np.isfinite(y_data_full)
            if not np.all(valid_indices):
                 print(f"WARNING: Non-finite values detected in '{output_name}'. Filtering again.")
                 X_data_current = X_data_full[valid_indices, :]
                 y_data_current = y_data_full[valid_indices]
            else:
                X_data_current = X_data_full
                y_data_current = y_data_full

            n_current = len(y_data_current)
            if n_current < 10:
                print(f"  ERROR: Not enough finite data points ({n_current}) for '{output_name}'. Skipping model.")
                summary_text += f"\n\n{output_name}: Insufficient data ({n_current} points)."
                continue # Skip to next output

            train_idx, test_idx = split_train_test_indices(n_current, test_frac=TEST_FRACTION, seed=RANDOM_SEED)
            X_train, y_train = X_data_current[train_idx, :], y_data_current[train_idx]
            X_test, y_test = X_data_current[test_idx, :], y_data_current[test_idx]
            print(f"  Data split: {len(train_idx)} train samples, {len(test_idx)} test samples.")

            try:
                surrogate_model_bundle = kriging_model.fit_kriging_model(X_train, y_train)

                model_filename = MODEL_FILENAME_TEMPLATE.format(output_name=output_name)
                model_path = os.path.join(casePath, model_filename)
                kriging_model.save_kriging_model(surrogate_model_bundle, model_path)
                print(f"  Model for {output_name} saved successfully.")

                y_pred_train, _ = kriging_model.eval_kriging_model(surrogate_model_bundle, X_train)
                y_pred_test, _ = kriging_model.eval_kriging_model(surrogate_model_bundle, X_test)

                metrics_train = calculate_metrics(y_train, y_pred_train)
                metrics_test = calculate_metrics(y_test, y_pred_test)

                print(f" Model Performance ({output_name}):")
                print(f" Train Set: RMSE={metrics_train['rmse']:.4f}, MAE={metrics_train['mae']:.4f}, R2={metrics_train['r2']:.4f}")
                print(f" Test Set:  RMSE={metrics_test['rmse']:.4f}, MAE={metrics_test['mae']:.4f}, R2={metrics_test['r2']:.4f}")

                summary_text += (
                    f"\n\n{output_name}:\n"
                    f"  Train: RMSE={metrics_train['rmse']:.4f}, MAE={metrics_train['mae']:.4f}, R²={metrics_train['r2']:.4f}\n"
                    f"  Test:  RMSE={metrics_test['rmse']:.4f}, MAE={metrics_test['mae']:.4f}, R²={metrics_test['r2']:.4f}"
                )
                current_y_pos -= 0.1 

                fig_pred, ax_pred = plt.subplots(figsize=(7, 7))
                ax_pred.scatter(y_train, y_pred_train, alpha=0.6, label=f"Train (N={len(y_train)}, R²={metrics_train['r2']:.3f})", s=30)
                ax_pred.scatter(y_test, y_pred_test, alpha=0.8, marker='s', label=f"Test (N={len(y_test)}, R²={metrics_test['r2']:.3f})", s=40)

                all_y = np.concatenate([y_train, y_pred_train, y_test, y_pred_test])
                valid_y = all_y[np.isfinite(all_y)]
                if len(valid_y) > 0:
                    min_val, max_val = np.min(valid_y), np.max(valid_y)
                    range_val = max_val - min_val if max_val > min_val else 1.0
                    lims = [min_val - 0.05 * range_val, max_val + 0.05 * range_val]
                    ax_pred.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x')
                    ax_pred.set_xlim(lims)
                    ax_pred.set_ylim(lims)
                else:
                     ax_pred.plot([0, 1], [0, 1], 'k--', alpha=0.75, zorder=0, label='y=x') 

                ax_pred.set_xlabel(f"True {output_name}")
                ax_pred.set_ylabel(f"Kriging Predicted {output_name}")
                ax_pred.set_title(f"Predicted vs. True Values - {output_name}")
                ax_pred.legend(fontsize=9)
                ax_pred.grid(True, linestyle='--', alpha=0.5)
                pdf.savefig(fig_pred, bbox_inches="tight")
                plt.close(fig_pred)

            except Exception as e:
                print(f"  ERROR: Failed during processing for {output_name}. Error: {e}")
                summary_text += f"\n\n{output_name}: FAILED - {e}"
                current_y_pos -= 0.05 

        ax_summary.text(0.05, current_y_pos + 0.1, summary_text, va="top", family="monospace", fontsize=9)
        pdf.savefig(fig_summary, bbox_inches="tight")
        plt.close(fig_summary)

if __name__ == "__main__":
    main()

