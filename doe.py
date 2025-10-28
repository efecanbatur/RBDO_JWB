import os
import csv
import numpy as np
import pandas as pd
import re
import subprocess
import sys
from pyDOE3 import lhs

# Import the merged simulation engine (which contains all necessary functions)
import VSP_MachLine as simulation_engine

# --- 1. DOE Configuration ---

# Total number of samples to run (MUST BE >= 1 to include baseline)
N_SAMPLES = 100  # <-- *** SET YOUR TOTAL SAMPLE SIZE HERE ***

# Output CSV file name
OUTPUT_CSV = "rbdo_dataset.csv"

# --- 2. Variable Definitions ---

# Define the 12 GEOMETRY (Design) Variables to sample
# Format: "variable_name": (min_bound, max_bound)
DESIGN_VARIABLES = {
    "W_root_chord_0": (18.2, 24.6),  # Default: 21.43
    "W_root_chord_1": (6.8, 9.2),    # Default: 8.043
    "W_root_chord_2": (3.7, 5.0),    # Default: 4.389
    "W_tip_chord_0": (6.8, 9.2),     # Default: 8.043
    "W_tip_chord_1": (3.7, 5.0),     # Default: 4.389
    "W_tip_chord_2": (1.4, 2.0),     # Default: 1.716
    "W_span_0": (1.0, 1.5),          # Default: 1.264
    "W_span_1": (1.0, 1.5),          # Default: 1.263
    "W_span_2": (1.7, 2.4),          # Default: 2.051
    "W_sweep_0": (80.0, 90.0),       # Default: 85.0
    "W_sweep_1": (70.0, 80.0),       # Default: 75.0
    "W_sweep_2": (65.0, 75.0)        # Default: 70.0
}
# Keep a fixed order for Design Variables
DESIGN_VAR_NAMES_ORDERED = list(DESIGN_VARIABLES.keys())

# Define the 2 UNCERTAIN (Flight Condition) Variables to sample
UNCERTAIN_VARIABLES = {
    "M": (1.4, 1.8),            # Default: 1.6
    "alpha_deg": (1.0, 5.0)     # Default: 3.06
}
UNCERTAIN_VAR_NAMES_ORDERED = ["M", "alpha_deg"]

# --- 3. Baseline Value Definition ---
# Get the baseline DV values from the simulation engine
default_dv_dict = simulation_engine.get_default_dv()
BASELINE_DVS = {key: default_dv_dict[key] for key in DESIGN_VAR_NAMES_ORDERED}

# Define baseline UV values (from run_full_case defaults)
BASELINE_UVS = {
    "M": 1.6,
    "alpha_deg": 3.06
}

# Combine baseline DVs and UVs
BASELINE_VALUES_DICT = {**BASELINE_DVS, **BASELINE_UVS}

# --- 4. Main DOE Generation and Execution ---

def main():
    if N_SAMPLES < 1:
         print("ERROR: N_SAMPLES must be 1 or greater to include the baseline.")
         sys.exit(1)

    # Combine all variables for sampling bounds definition
    all_variables_info = {**DESIGN_VARIABLES, **UNCERTAIN_VARIABLES}
    # Create the specific list of variables for sampling order
    sampled_var_names = DESIGN_VAR_NAMES_ORDERED + UNCERTAIN_VAR_NAMES_ORDERED
    n_dim = len(sampled_var_names)

    # Get min/max bounds corresponding to the sampled_var_names order
    min_bounds = np.array([all_variables_info[key][0] for key in sampled_var_names])
    max_bounds = np.array([all_variables_info[key][1] for key in sampled_var_names])

    print(f"--- RBDO DOE Generation ---")
    print(f"Generating {N_SAMPLES} samples for {n_dim} dimensions (including baseline at index 0)...")
    print(f"Sampled Variables (in internal order): {sampled_var_names}")

    # --- Sample Generation with Baseline ---
    # Create the array to hold all samples
    doe_samples_scaled = np.zeros((N_SAMPLES, n_dim))

    # Set the first row (index 0) to the baseline values in the sampling order
    baseline_values_ordered_for_sampling = [BASELINE_VALUES_DICT[key] for key in sampled_var_names]
    doe_samples_scaled[0, :] = baseline_values_ordered_for_sampling
    print("Sample 0 set to baseline values.")

    # Generate the remaining N_SAMPLES - 1 samples using LHS if needed
    if N_SAMPLES > 1:
        n_lhs_samples = N_SAMPLES - 1
        print(f"Generating {n_lhs_samples} LHS samples...")
        # Generate LHS samples in the [0, 1] hypercube
        doe_samples_01 = lhs(n_dim, samples=n_lhs_samples, criterion='maximin')
        # Scale samples to their physical bounds
        lhs_samples_scaled = min_bounds + doe_samples_01 * (max_bounds - min_bounds)
        # Place the LHS samples into the array starting from the second row (index 1)
        doe_samples_scaled[1:, :] = lhs_samples_scaled
    # --- End Sample Generation ---

    # Create a DataFrame using the sampled variables order (easier access by name)
    df = pd.DataFrame(doe_samples_scaled, columns=sampled_var_names)

    results_rows = []

    # --- Define CSV headers IN THE DESIRED OUTPUT ORDER ---
    headers = ["sample_id"] + UNCERTAIN_VAR_NAMES_ORDERED + \
              ["CL", "CD", "PLdB"] + DESIGN_VAR_NAMES_ORDERED

    # Get the full default DV dictionary (used for non-sampled variables)
    base_dv_dict = simulation_engine.get_default_dv()

    # Loop through each sample row (including the baseline at index 0)
    for idx, row in df.iterrows():
        print(f"\n--- Running Sample {idx} / {N_SAMPLES-1} ---")

        # 1. Create the unique DV dictionary for this sample
        current_dv_dict = base_dv_dict.copy()
        for key in DESIGN_VAR_NAMES_ORDERED:
            current_dv_dict[key] = row[key] # Overwrite with sampled or baseline value

        # 2. Extract the unique UV dictionary for this sample
        current_uv_dict = {
            "M": row["M"],
            "alpha_deg": row["alpha_deg"]
        }

        # 3. Define a unique casename for this sample
        casename = f"rbdo_sample_{idx}"

        CL_val, CD_val, pldb_val = np.nan, np.nan, np.nan # Initialize results as NaN

        try:
            # --- STEP 4: Build Geometry ---
            print(f"Building geometry: {casename}.stl")
            _ , stl_path = simulation_engine.objective_Function(current_dv_dict, casename=casename)

            # --- STEP 5: Run MachLine ---
            print(f"Running MachLine...")
            freestream = simulation_engine.calculate_freestream(V_inf=500, aoa=current_uv_dict['alpha_deg'], beta=0)
            coeffs = simulation_engine.run_machline_analysis(
                stl_save_path=stl_path,
                freestream=freestream,
                machNumber=current_uv_dict['M'],
                Sref=65.5,
                formulation="dirichlet-morino",
                casename=casename
            )
            CL_val = coeffs.get("Cz", np.nan)
            CD_val = coeffs.get("Cx", np.nan)

            # --- STEP 6: Run ITUBOOM ---
            print(f"Running ITUBOOM...")
            try:
                result = subprocess.run(
                    [sys.executable, "example_1.py", str(current_uv_dict['M']), casename, str(current_uv_dict['alpha_deg'])],
                    cwd=simulation_engine.casePath_ituboom,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                stdout_text = result.stdout
                print(stdout_text)
                # Extract PLdB logic (copied from VSP_MachLine.run_full_case)
                num = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
                patterns = [rf'PLdB\s*[:=]?\s*{num}', rf'Perceived\s+Loudness\s*[:=]\s*{num}', rf'{num}\s*dB\b']
                pldb_val_search = None
                for pat in patterns:
                    m = re.search(pat, stdout_text, flags=re.IGNORECASE)
                    if m:
                        pldb_val_search = float(m.group(1)); break
                if pldb_val_search is None:
                    floats = re.findall(num, stdout_text)
                    if floats: pldb_val_search = float(floats[-1])
                if pldb_val_search is None:
                    print(f"WARNING: Could not parse PLdB for {casename}. Setting to NaN.")
                    pldb_val = np.nan
                else:
                    pldb_val = pldb_val_search
            except Exception as ituboom_e:
                print(f"ERROR: ITUBOOM subprocess failed for sample {idx}: {ituboom_e}")
                pldb_val = np.nan

            print(f"--- Sample {idx} Complete: CL={CL_val:.4f}, CD={CD_val:.4f}, PLdB={pldb_val:.2f} ---")

        except Exception as e:
            # Catch errors during Geometry or MachLine stages
            print(f"ERROR: A critical error occurred before or during MachLine for sample {idx}: {e}")
            # CL, CD, PLdB remain NaN as initialized

        # --- 7. Store results IN THE DESIRED CSV ORDER ---
        uv_values = [row["M"], row["alpha_deg"]]
        output_values = [CL_val, CD_val, pldb_val]
        dv_values = [row[key] for key in DESIGN_VAR_NAMES_ORDERED]
        # Append row data matching the header order
        results_rows.append([idx] + uv_values + output_values + dv_values)

    # 8. Save final DOE table as csv
    results_file = os.path.join(simulation_engine.casePath, OUTPUT_CSV)
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers) # Write the correctly ordered headers
        writer.writerows(results_rows) # Write the correctly ordered row data

    print("\n\n" + "="*50)
    print("All RBDO DOE evaluations completed.")
    print(f"Dataset saved to: {results_file}")
    print("="*50)

if __name__ == "__main__":
    main()

