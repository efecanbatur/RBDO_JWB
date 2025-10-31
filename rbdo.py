#     find the Pareto optimal set of solutions for:
#     Objective 1: Minimize mean(PLdB)
#     Objective 2: Minimize mean(CD)
#     Constraint 1: Prob(PLdB > 91 dB) <= 10%
#     Constraint 2: mean(CL) >= 0.06


import os
import json
import numpy as np
import pce_model
import input_models

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize as pymoo_minimize
except ImportError:
    print("Error: pymoo is required for this script. Please install it using 'pip install pymoo'")
    Problem, NSGA2, get_termination, pymoo_minimize = (None,) * 4

# pathlines + config
casePath = os.path.dirname(os.path.abspath(__file__))
pce_model_pldb_path = os.path.join(casePath, "pce_model_pldb.json")
pce_model_cl_path = os.path.join(casePath, "pce_model_cl.json")
pce_model_cd_path = os.path.join(casePath, "pce_model_cd.json") 
input_model_path = os.path.join(casePath, "input_model.json")

# RBDO definition
DV_BOUNDS = [(1.5, 2.5), (1.7, 5.0)]
RELIABILITY_THRESHOLD = 91.0
RELIABILITY_TARGET_PF = 0.10
TARGET_CL_MIN = 0.060
N_SAMPLES_MC = 10000
CACHE = {}

# utility functions
def _ensure_exists(path, msg):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg} not found at: {path}")



def modify_input_model_nominals(x_dv):
    with open(input_model_path, "r") as f:
        cfg = json.load(f)
    params = cfg["aleatory"]["params"]
    params[0]["mu"], params[1]["mu"] = float(x_dv[0]), float(x_dv[1])
    tmp_path = os.path.join(casePath, "_tmp_input_model_rbdo.json")
    with open(tmp_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return tmp_path


# unified evaluation function - inner loop
def evaluate_performance(x_dv_tuple, pce_pldb, pce_cl, pce_cd):
    if x_dv_tuple in CACHE:
        return CACHE[x_dv_tuple]
    x_dv = np.array(x_dv_tuple)
    print(f"  > Evaluating design: M_nom={x_dv[0]:.4f}, AoA_nom={x_dv[1]:.4f}")
    
    tmp_input_path = modify_input_model_nominals(x_dv)
    Xs, names = input_models.sample_aleatory(tmp_input_path, n=N_SAMPLES_MC, seed=42)
    samples = np.column_stack([Xs[:, names.index("M")], Xs[:, names.index("alpha_deg")]])
    os.remove(tmp_input_path)
    
    # eval surrogates
    pldb_preds = pce_model.eval_pce_model(pce_pldb, samples)
    cl_preds = pce_model.eval_pce_model(pce_cl, samples)
    cd_preds = pce_model.eval_pce_model(pce_cd, samples)
    
    # calculate metrics
    mean_pldb = np.mean(pldb_preds)
    mean_cl = np.mean(cl_preds)
    mean_cd = np.mean(cd_preds)
    pf_pldb = np.sum(pldb_preds > RELIABILITY_THRESHOLD) / N_SAMPLES_MC # pf = probability of failure
    
    result = (mean_pldb, mean_cl, mean_cd, pf_pldb)
    CACHE[x_dv_tuple] = result
    return result

# problem definition - multi-objective pymoo
class RDBOMultiObjectiveProblem(Problem):
    def __init__(self, pce_pldb, pce_cl, pce_cd):
        super().__init__(n_var=2, n_obj=2, n_constr=2, # 2 objectives, 2 constraints
                         xl=np.array([b[0] for b in DV_BOUNDS]),
                         xu=np.array([b[1] for b in DV_BOUNDS]))
        self.pce_pldb = pce_pldb
        self.pce_cl = pce_cl
        self.pce_cd = pce_cd

    def _evaluate(self, x, out, *args, **kwargs):
        objectives, constraints = [], []
        for design_point in x:
            mean_pldb, mean_cl, mean_cd, pf_pldb = evaluate_performance(
                tuple(design_point), self.pce_pldb, self.pce_cl, self.pce_cd
            )
            
            # objectives [min PLdB, min CD]
            objectives.append([mean_pldb, mean_cd])
            
            # constraints: must be <= 0 for feasibility
            # constraint 1: pf - target <= 0
            g1 = pf_pldb - RELIABILITY_TARGET_PF
            # constraint 2: target_CL - mean_CL <= 0
            g2 = TARGET_CL_MIN - mean_cl
            constraints.append([g1, g2])

        out["F"] = np.array(objectives)
        out["G"] = np.array(constraints)

# main RBDO function block
def main():
    if not Problem:
        raise ImportError("pymoo is required for this script. Please install it using 'pip install pymoo'")

    print("Multi-Objective RBDO with NSGA-II (PLdB, CD objective; CL, Reliability constraint)")
    
    _ensure_exists(pce_model_pldb_path, "PLdB PCE model")
    _ensure_exists(pce_model_cl_path, "CL PCE model")
    _ensure_exists(pce_model_cd_path, "CD PCE model") 
    _ensure_exists(input_model_path, "Input model")

    # load all PCE models
    pce_pldb = pce_model.load_pce_model(pce_model_pldb_path)
    pce_cl = pce_model.load_pce_model(pce_model_cl_path)
    pce_cd = pce_model.load_pce_model(pce_model_cd_path)

    print(f"\nStarting NSGA-II Optimization")

    problem = RDBOMultiObjectiveProblem(pce_pldb, pce_cl, pce_cd)
    algorithm = NSGA2(pop_size=40)
    termination = get_termination("n_gen", 40)
    
    result = pymoo_minimize(problem, algorithm, termination, seed=1, verbose=True)

    # print results
    print("\nRBDO Complete")
    if result is None:
        print("Optimization did not run successfully.")
        return

    print("Multi-objective optimization finished.")
    print("\nPareto Front:")
    print("-------------------------------------------------------------------------------------------------")
    print(f"{'#':>3} {'Nominal M':>12} {'Nominal AoA':>14} {'Mean PLdB':>14} {'Mean CD':>12} {'Mean CL':>12} {'PLdB Pf':>10}")
    print("-------------------------------------------------------------------------------------------------")
    
    if result.X is None:
        print("No feasible solutions were found.")
        return
        
    for i, (design, objective) in enumerate(zip(result.X, result.F)):
        # re-eval
        _, final_mean_cl, _, final_pf_pldb = evaluate_performance(tuple(design), pce_pldb, pce_cl, pce_cd)
        print(f"{i+1:3d} {design[0]:12.4f} {design[1]:14.4f} {objective[0]:14.4f} {objective[1]:12.6f} {final_mean_cl:12.4f} {final_pf_pldb:10.4f}")

if __name__ == "__main__":
    main()

