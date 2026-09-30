import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score
from ucimlrepo import fetch_ucirepo

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("="*60)
print(" SPEA2 Feature Selection Demo - Internet Advertisements")
print("="*60)

# 1. Load Dataset
print("\n[1/5] Loading Internet Advertisements dataset via ucimlrepo...")
start_time = time.time()
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"
    df = pd.read_csv(url, header=None, low_memory=False)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
except Exception as e:
    print(f"Error downloading dataset directly from UCI: {e}")
    exit(1)

print(f"Original dataset shape: Features(X)={X.shape}, Targets(y)={y.shape}")
print(f"Time taken to load data: {time.time() - start_time:.2f}s")

# 2. Preprocess Dataset
print("\n[2/5] Preprocessing data...")
# Handle missing values by coercing to numeric and filling with column mean
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# Ensure y is 1D string array and convert to binary classification: 'ad.' vs 'nonad.'
y_series = y.iloc[:, 0].astype(str).str.strip()
y_binary = np.where(y_series == 'ad.', 1, 0)
class_dist = pd.Series(y_binary).value_counts(normalize=True) * 100
print(f"Class Distribution: Non-Ad (0) = {class_dist[0]:.2f}%, Ad (1) = {class_dist[1]:.2f}%")

# Convert to fast numpy arrays for cross-validation
X_np = X.values
y_np = y_binary

# 3. Define Evaluation Setup
print("\n[3/5] Initializing Evaluation Metrics and Classifier...")
# We use DecisionTree for fast evaluation during the demo
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='binary')

def evaluate_features(individual):
    """
    Evaluates a candidate solution (chromosome).
    Returns (F1_Score, Num_Features_Selected)
    """
    # Convert binary chromosome to boolean mask
    features_mask = np.array(individual, dtype=bool)
    num_features = np.sum(features_mask)
    
    # Penalty if no features selected
    if num_features == 0:
        return 0.0, X_np.shape[1]
    
    # Subset the features
    X_subset = X_np[:, features_mask]
    
    # Objective 1: Maximize F1-Score (robust to imbalance)
    scores = cross_val_score(clf, X_subset, y_np, cv=cv, scoring=f1_scorer, n_jobs=-1)
    f1 = scores.mean()
    
    # Objective 2: Minimize Number of Features
    return f1, num_features

# 4. Configure DEAP for SPEA2
print("\n[4/5] Configuring DEAP framework for SPEA2...")
# Fitness: (Maximize F1, Minimize Feature Count)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
NUM_FEATURES = X.shape[1]

# Feature genes: 0 (exclude) or 1 (include)
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, NUM_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_features)
toolbox.register("mate", tools.cxTwoPoint)
# Mutate with a probability such that roughly 1 bit is flipped per individual (1 / 1558)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/NUM_FEATURES) 
toolbox.register("select", tools.selSPEA2)

# 5. Run SPEA2 Algorithm
def run_spea2():
    # Parameters for the demo (keep small for fast execution)
    POP_SIZE = 30
    GENS = 5
    ARCHIVE_SIZE = 15
    
    pop = toolbox.population(n=POP_SIZE)
    archive = []
    
    print(f"\n[5/5] Starting SPEA2 Evolution (Pop Size: {POP_SIZE}, Generations: {GENS})...")
    start_evol = time.time()
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for gen in range(GENS):
        gen_start = time.time()
        
        # SPEA2 Selection uses environmental selection on pop + archive
        # It maintains an archive of fixed size, computing raw fitness and density
        archive = toolbox.select(pop + archive, k=POP_SIZE)
        pop = archive
        
        # Create offspring
        offspring = list(map(toolbox.clone, pop))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8: # cxpb
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < 0.2: # mutpb
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate valid offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Replace population
        pop[:] = offspring
        
        # Extract best F1 so far from archive for logging
        best_f1 = max([ind.fitness.values[0] for ind in archive]) if archive else 0.0
        print(f"  -> Generation {gen+1}/{GENS} completed in {time.time()-gen_start:.2f}s | Best Archive F1: {best_f1:.4f}")
        
    print(f"\nEvolution complete in {time.time()-start_evol:.2f}s!")
    
    # Extract Pareto Front
    pareto_front = tools.sortNondominated(archive, len(archive), first_front_only=True)[0]
    
    print(f"\nFound {len(pareto_front)} non-dominated solutions in the Pareto front.")
    
    # Display Pareto Front results
    print("\nPareto Front (Trade-off options):")
    print(f"{'F1-Score':<15} | {'Feature Count':<15} | {'% Reduction':<15}")
    print("-" * 50)
    
    # Sort by feature count for clean display
    pareto_front.sort(key=lambda x: x.fitness.values[1])
    
    f1_scores = []
    feature_counts = []
    
    for ind in pareto_front:
        f1 = ind.fitness.values[0]
        count = int(ind.fitness.values[1])
        reduction = (1 - count/NUM_FEATURES) * 100
        
        f1_scores.append(f1)
        feature_counts.append(count)
        print(f"{f1:<15.4f} | {count:<15d} | {reduction:<15.2f}%")
        
    # Plotting
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(feature_counts, f1_scores, c='blue', marker='o', alpha=0.7)
        plt.plot(feature_counts, f1_scores, c='blue', linestyle='--', alpha=0.3)
        plt.title('SPEA2 Pareto Front: Accuracy vs Feature Count')
        plt.xlabel('Number of Features Selected')
        plt.ylabel('F1-Score (Ad Detection)')
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Highlight best accuracy and best feature reduction
        if feature_counts:
            best_acc_idx = np.argmax(f1_scores)
            plt.scatter(feature_counts[best_acc_idx], f1_scores[best_acc_idx], c='green', s=100, label='Highest F1', zorder=5)
            best_feat_idx = np.argmin(feature_counts)
            plt.scatter(feature_counts[best_feat_idx], f1_scores[best_feat_idx], c='red', s=100, label='Fewest Features', zorder=5)
            plt.legend()
            
        plt.tight_layout()
        plt.savefig('spea2_pareto_front.png')
        print("\nSaved Pareto front visualization to 'spea2_pareto_front.png'")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

if __name__ == "__main__":
    run_spea2()
