import os
import csv
import numpy as np
import pandas as pd
import re
import subprocess
import sys
from pyDOE3 import lhs
import VSP_MachLine as simulation_engine

N_SAMPLES = 100  # doe size, change accordingly
OUTPUT_CSV = "rbdo_dataset.csv"

# format: "variable_name": (min_bound, max_bound)
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
# keep a fixed order for design variables
DESIGN_VAR_NAMES_ORDERED = list(DESIGN_VARIABLES.keys())

# define 2 uncertain variables to sample
UNCERTAIN_VARIABLES = {
    "M": (1.4, 1.8),            # Default: 1.6
    "alpha_deg": (0.0, 6.12)     # Default: 3.06
}
UNCERTAIN_VAR_NAMES_ORDERED = ["M", "alpha_deg"]

# baseline definition
# import baseline DV values from the simulation engine
default_dv_dict = simulation_engine.get_default_dv()
BASELINE_DVS = {key: default_dv_dict[key] for key in DESIGN_VAR_NAMES_ORDERED}

BASELINE_UVS = {
    "M": 1.6,
    "alpha_deg": 3.06
}
BASELINE_VALUES_DICT = {**BASELINE_DVS, **BASELINE_UVS}

# doe generation

def main():
    if N_SAMPLES < 1:
         print("ERROR: N_SAMPLES must be 1 or greater to include the baseline.")
         sys.exit(1)

    # combine all variables for sampling bounds definition
    all_variables_info = {**DESIGN_VARIABLES, **UNCERTAIN_VARIABLES}
    # create list of variables for sampling order
    sampled_var_names = DESIGN_VAR_NAMES_ORDERED + UNCERTAIN_VAR_NAMES_ORDERED
    n_dim = len(sampled_var_names)

    # get min/max bounds corresponding to the sampled_var_names order
    min_bounds = np.array([all_variables_info[key][0] for key in sampled_var_names])
    max_bounds = np.array([all_variables_info[key][1] for key in sampled_var_names])

    print(f"RBDO DOE Generation")
    print(f"Generating {N_SAMPLES} samples for {n_dim} dimensions...")
    print(f"Sampled Variables: {sampled_var_names}")
    doe_samples_scaled = np.zeros((N_SAMPLES, n_dim))

    # set the first row to the baseline values in the sampling order
    baseline_values_ordered_for_sampling = [BASELINE_VALUES_DICT[key] for key in sampled_var_names]
    doe_samples_scaled[0, :] = baseline_values_ordered_for_sampling
    print("Sample 0 set to baseline values.")

    # generate the remaining N_SAMPLES - 1 samples using LHS
    if N_SAMPLES > 1:
        n_lhs_samples = N_SAMPLES - 1
        print(f"Generating {n_lhs_samples} LHS samples...")
        # generate LHS samples in the [0, 1] hypercube
        doe_samples_01 = lhs(n_dim, samples=n_lhs_samples, criterion='maximin')
        # scale samples to their physical bounds
        lhs_samples_scaled = min_bounds + doe_samples_01 * (max_bounds - min_bounds)
        # place the LHS samples into the array starting from the second row
        doe_samples_scaled[1:, :] = lhs_samples_scaled

    # create a DataFrame using the sampled variables order
    df = pd.DataFrame(doe_samples_scaled, columns=sampled_var_names)

    results_rows = []

    # define CSV headers in order
    headers = ["sample_id"] + UNCERTAIN_VAR_NAMES_ORDERED + \
              ["CL", "CD", "PLdB"] + DESIGN_VAR_NAMES_ORDERED

    # get the full default DV dictionary - for non-sampled variables
    base_dv_dict = simulation_engine.get_default_dv()

    # loop through each sample row
    for idx, row in df.iterrows():
        print(f"\nRunning Sample {idx} / {N_SAMPLES-1}")

        # create the unique DV dictionary for this sample
        current_dv_dict = base_dv_dict.copy()
        for key in DESIGN_VAR_NAMES_ORDERED:
            current_dv_dict[key] = row[key] # overwrite with sampled or baseline value

        # extract the unique UV dictionary for this sample
        current_uv_dict = {
            "M": row["M"],
            "alpha_deg": row["alpha_deg"]
        }

        # define a unique casename for this sample
        casename = f"rbdo_sample_{idx}"

        CL_val, CD_val, pldb_val = np.nan, np.nan, np.nan # initialize results as NaN

        try:
            # build geometry
            print(f"Building geometry: {casename}.stl")
            _ , stl_path = simulation_engine.objective_Function(current_dv_dict, casename=casename)

            # run MachLine
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

            # run ituboom
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
                # extract PLdB
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

                    #catch errors
            except Exception as ituboom_e:
                print(f"ERROR: ITUBOOM subprocess failed for sample {idx}: {ituboom_e}")
                pldb_val = np.nan

            print(f"--- Sample {idx} Complete: CL={CL_val:.4f}, CD={CD_val:.4f}, PLdB={pldb_val:.2f} ---")

        except Exception as e:

            # catch errors
            print(f"ERROR: An error occurred before or during MachLine for sample {idx}: {e}")
            # CL, CD, PLdB remain NaN as initialized

        # store results in csv in order
        uv_values = [row["M"], row["alpha_deg"]]
        output_values = [CL_val, CD_val, pldb_val]
        dv_values = [row[key] for key in DESIGN_VAR_NAMES_ORDERED]
        # append row data matching the header order
        results_rows.append([idx] + uv_values + output_values + dv_values)

    # save doe table as csv
    results_file = os.path.join(simulation_engine.casePath, OUTPUT_CSV)
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers) # write the correctly ordered headers
        writer.writerows(results_rows) # write the correctly ordered row data

    print("\n\n" + "="*50)
    print("All RBDO DOE evaluations completed.")
    print(f"Dataset saved to: {results_file}")
    print("="*50)

if __name__ == "__main__":
    main()

