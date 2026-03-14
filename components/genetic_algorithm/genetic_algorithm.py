import argparse
import os
import pandas as pd
import numpy as np
import time
import random
import warnings
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_target", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--n_gen", type=int, default=15)
    return parser.parse_args()

def load_data(path):
    """Load parquet file"""
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "train_features_filtered.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Could not find data in {path}")

def load_target(path):
    """Load target CSV"""
    if os.path.isdir(path):
        csv_path = os.path.join(path, "train_target.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)['RUL'].values
    raise FileNotFoundError(f"Could not find target in {path}")

def main():
    args = parse_args()
    
    print("Loading filtered features and target...")
    train_features = load_data(args.train_features)
    train_target = load_target(args.train_target)
    
    X_train = train_features.values
    n_features = X_train.shape[1]
    feature_names = train_features.columns.tolist()
    
    print(f"Features: {n_features}, Samples: {X_train.shape[0]}")
    
    # Define fitness and individual
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # Fitness function
    def evaluate(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) == 0:
            return (1e6,)
        
        X_sub = X_train[:, selected]
        rf = RandomForestRegressor(n_estimators=20, max_samples=0.3, random_state=42, n_jobs=-1)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        neg_mse = cross_val_score(rf, X_sub, train_target, cv=kf, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-neg_mse.mean())
        
        penalty = (len(selected) / n_features) * 0.05
        fitness = rmse + penalty
        
        return (fitness,)
    
    # DEAP setup
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: 1 if random.random() < 0.5 else 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=2)
    
    random.seed(42)
    np.random.seed(42)
    
    print(f"Running GA: pop={args.pop_size}, gen={args.n_gen}")
    start = time.time()
    
    pop = toolbox.population(n=args.pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    hof = tools.HallOfFame(1)
    
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=args.n_gen,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    elapsed = time.time() - start
    
    # Get best individual
    best = hof[0]
    best_idx = [i for i, bit in enumerate(best) if bit == 1]
    best_names = [feature_names[i] for i in best_idx]
    
    print(f"✅ GA complete in {elapsed:.1f}s")
    print(f"Best fitness: {best.fitness.values[0]:.4f}")
    print(f"Features selected: {len(best_idx)} / {n_features}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    pd.DataFrame({'feature': best_names}).to_csv(os.path.join(args.output_dir, "ga_selected_features.csv"), index=False)
    pd.DataFrame({'index': best_idx}).to_csv(os.path.join(args.output_dir, "ga_feature_indices.csv"), index=False)
    
    # Save metrics
    metrics = {
        "best_fitness": float(best.fitness.values[0]),
        "n_features_selected": len(best_idx),
        "n_features_total": n_features,
        "execution_time_seconds": elapsed,
        "population_size": args.pop_size,
        "generations": args.n_gen
    }
    
    import json
    with open(os.path.join(args.output_dir, "ga_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()