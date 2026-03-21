import argparse
import os
import time
import json
import random
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import deap
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="DEAP Genetic Algorithm for feature selection")
    parser.add_argument("--input_train_features", type=str, required=True)
    parser.add_argument("--input_labels",          type=str, required=True)
    parser.add_argument("--input_test_features",   type=str, required=True)
    parser.add_argument("--output_train_features", type=str, required=True)
    parser.add_argument("--output_test_features",  type=str, required=True)
    parser.add_argument("--output_ga_report",      type=str, required=True,
                        help="Output folder: GA run report JSON")
    # GA hyperparameters
    parser.add_argument("--population_size",   type=int,   default=50)
    parser.add_argument("--n_generations",     type=int,   default=30)
    parser.add_argument("--cx_prob",           type=float, default=0.7,
                        help="Crossover probability")
    parser.add_argument("--mut_prob",          type=float, default=0.2,
                        help="Mutation probability")
    parser.add_argument("--tournament_size",   type=int,   default=3)
    parser.add_argument("--alpha",             type=float, default=0.01,
                        help="Penalty weight for feature count in fitness")
    parser.add_argument("--cv_folds",          type=int,   default=3,
                        help="Cross-validation folds for fitness evaluation")
    parser.add_argument("--random_seed",       type=int,   default=42)
    return parser.parse_args()


def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder_path}")
    return pd.concat(
        [pd.read_parquet(os.path.join(folder_path, f)) for f in files],
        ignore_index=True
    )


# ── DEAP fitness function ──────────────────────────────────────────────────────

def evaluate(individual, X: np.ndarray, y: np.ndarray,
             alpha: float, cv_folds: int):
    """
    Fitness = -RMSE  -  alpha * feature_ratio
    Higher is better (DEAP maximises by default when we set weights=(1.0,)).
    A chromosome is a binary vector of length n_features.
    """
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

    # Penalise empty chromosomes heavily
    if len(selected_idx) == 0:
        return (-1e6,)

    X_sel = X[:, selected_idx]
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        n_jobs=-1,
        random_state=42
    )
    neg_mse_scores = cross_val_score(
        rf, X_sel, y,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=1
    )
    rmse = np.sqrt(-neg_mse_scores.mean())
    feature_ratio = len(selected_idx) / len(individual)
    fitness = -rmse - alpha * feature_ratio
    return (fitness,)


def run_ga(X: np.ndarray, y: np.ndarray, n_features: int,
           pop_size: int, n_gen: int, cx_prob: float, mut_prob: float,
           tourn_size: int, alpha: float, cv_folds: int,
           seed: int) -> tuple:
    """Run DEAP GA; return (best_individual, log_df)."""
    random.seed(seed)
    np.random.seed(seed)

    # Avoid duplicate class registration across repeated imports
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate,
                     X=X, y=y, alpha=alpha, cv_folds=cv_folds)
    toolbox.register("mate",    tools.cxTwoPoint)
    toolbox.register("mutate",  tools.mutFlipBit, indpb=0.05)
    toolbox.register("select",  tools.selTournament, tournsize=tourn_size)

    population = toolbox.population(n=pop_size)

    # Hall-of-fame (single best individual)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max",  np.max)
    stats.register("mean", np.mean)
    stats.register("min",  np.min)

    print(f"  GA config: pop={pop_size}, gens={n_gen}, "
          f"cx={cx_prob}, mut={mut_prob}, alpha={alpha}, cv={cv_folds}")

    log = tools.Logbook()
    log.header = ["gen", "nevals", "max", "mean", "min"]

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(population)
    log.record(gen=0, nevals=len(population), **record)
    print(f"  Gen 0  | max={record['max']:.4f}  mean={record['mean']:.4f}")

    for gen in range(1, n_gen + 1):
        offspring = algorithms.varAnd(population, toolbox,
                                      cxpb=cx_prob, mutpb=mut_prob)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(offspring + population, k=pop_size)
        hof.update(population)

        record = stats.compile(population)
        log.record(gen=gen, nevals=len(invalid), **record)

        if gen % 5 == 0 or gen == n_gen:
            n_sel = sum(hof[0])
            print(f"  Gen {gen:3d} | max={record['max']:.4f}  "
                  f"mean={record['mean']:.4f}  "
                  f"best_features={n_sel}")

    log_df = pd.DataFrame(log)
    return list(hof[0]), log_df


def main():
    args = parse_args()
    start = time.time()

    print("=" * 60)
    print("COMPONENT: genetic_algorithm")
    print("=" * 60)

    # --- Load ---
    print("\n[1/3] Loading data ...")
    train_df  = load_parquet_folder(args.input_train_features)
    labels_df = load_parquet_folder(args.input_labels)
    test_df   = load_parquet_folder(args.input_test_features)

    meta_cols    = ["entity_id"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[[c for c in feature_cols if c in test_df.columns]].values.astype(np.float32)

    y_train = labels_df.set_index("entity_id")["target_RUL"].reindex(
        train_df["entity_id"]
    ).fillna(0).values.astype(np.float32)

    # Sanitise
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- Run GA ---
    print("\n[2/3] Running Genetic Algorithm ...")
    best_individual, log_df = run_ga(
        X=X_train, y=y_train,
        n_features=len(feature_cols),
        pop_size=args.population_size,
        n_gen=args.n_generations,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        tourn_size=args.tournament_size,
        alpha=args.alpha,
        cv_folds=args.cv_folds,
        seed=args.random_seed,
    )

    selected_idx  = [i for i, bit in enumerate(best_individual) if bit == 1]
    selected_cols = [feature_cols[i] for i in selected_idx]
    print(f"\n  GA selected {len(selected_cols)} / {len(feature_cols)} features")

    # --- Filter and save ---
    print("\n[3/3] Saving outputs ...")
    X_train_sel = X_train[:, selected_idx]
    X_test_sel  = X_test[:,  selected_idx]

    train_out = pd.DataFrame(X_train_sel, columns=selected_cols)
    train_out.insert(0, "entity_id", train_df["entity_id"].values)

    test_out  = pd.DataFrame(X_test_sel, columns=selected_cols)
    test_out.insert(0, "entity_id", test_df["entity_id"].values)

    os.makedirs(args.output_train_features, exist_ok=True)
    os.makedirs(args.output_test_features,  exist_ok=True)
    os.makedirs(args.output_ga_report,      exist_ok=True)

    train_out.to_parquet(os.path.join(args.output_train_features, "data.parquet"), index=False)
    test_out.to_parquet( os.path.join(args.output_test_features,  "data.parquet"), index=False)

    report = {
        "selected_feature_count": len(selected_cols),
        "total_input_features":   len(feature_cols),
        "selected_features":      selected_cols,
        "ga_config": {
            "population_size": args.population_size,
            "n_generations":   args.n_generations,
            "cx_prob":         args.cx_prob,
            "mut_prob":        args.mut_prob,
            "alpha":           args.alpha,
            "cv_folds":        args.cv_folds,
            "random_seed":     args.random_seed,
        },
        "evolution_log": log_df.to_dict(orient="records"),
    }
    with open(os.path.join(args.output_ga_report, "ga_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - start
    print(f"\n✅ genetic_algorithm complete in {elapsed:.1f}s")
    print(f"   Features selected: {len(selected_cols)}")
    print(f"   GA report → {args.output_ga_report}/ga_report.json")


if __name__ == "__main__":
    main()