import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("="*60)
print(" Real-Time Ad-Blocker Classification Demo")
print("="*60)

DATA_FILE = "ad_data_cache.csv"

# 1. Load Data (with local caching for speed)
if os.path.exists(DATA_FILE):
    print("\n[1/4] Loading dataset from local cache (fast)...")
    df = pd.read_csv(DATA_FILE, header=None, low_memory=False)
else:
    print("\n[1/4] Downloading dataset from UCI (this may take a few minutes)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"
    df = pd.read_csv(url, header=None, low_memory=False)
    df.to_csv(DATA_FILE, index=False, header=False)
    print("      Dataset cached locally for future runs!")

# 2. Preprocess Data
print("[2/4] Preprocessing image and URL features...")
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean()).values

y_series = y.iloc[:, 0].astype(str).str.strip()
y_binary = np.where(y_series == 'ad.', 1, 0)

# 3. Simulate GA Feature Reduction
print("[3/4] Applying optimized feature subset (Simulating SPEA2 output of ~730 features)...")
# (In a real system, this mask would be loaded directly from the SPEA2 output)
temp_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
temp_clf.fit(X, y_binary)
importances = temp_clf.feature_importances_
best_features_mask = np.argsort(importances)[-730:] 

X_optimized = X[:, best_features_mask]

# 4. Train Final Model
print("[4/4] Training the final real-time Ad-Blocker classifier...")
X_train, X_test, y_train, y_test = train_test_split(X_optimized, y_binary, test_size=0.1, random_state=42)

ad_blocker = DecisionTreeClassifier(random_state=42, max_depth=5)
ad_blocker.fit(X_train, y_train)


print("\n" + "="*60)
print(" [!] LIVE BROWSER AD-BLOCKING SIMULATION [!]")
print("="*60)
print("Simulating a browser evaluating new web page elements in real-time...\n")

# Pick a mix of ads and non-ads to demonstrate
ad_indices = np.where(y_test == 1)[0]
nonad_indices = np.where(y_test == 0)[0]
demo_indices = list(np.random.choice(ad_indices, 2, replace=False)) + list(np.random.choice(nonad_indices, 3, replace=False))
np.random.shuffle(demo_indices)

for i, idx in enumerate(demo_indices):
    sample = X_test[idx].reshape(1, -1)
    actual_label = y_test[idx]
    
    print(f"Scanning Element {i+1} / 5 ...")
    time.sleep(1.0) # Fake network delay for visual effect
    
    # --- REAL TIME PREDICTION ---
    start_time = time.time()
    prediction = ad_blocker.predict(sample)[0]
    pred_time_ms = (time.time() - start_time) * 1000
    # ----------------------------
    
    actual_type = "Advertisement" if actual_label == 1 else "Standard Web Content"
    
    print(f"  Ground Truth   : {actual_type}")
    if prediction == 1:
        print(f"  Action Taken   : [BLOCKED] Detected as Advertisement")
    else:
        print(f"  Action Taken   : [ALLOWED] Detected as Safe Content")
        
    print(f"  Processing Time: {pred_time_ms:.4f} ms")
    
    if actual_label == prediction:
        print("  Status         : [OK] Correct Classification")
    else:
        print("  Status         : [ERR] Misclassification")
    print("-" * 60)

print("\nDemo complete! Notice how the reduced feature set allows for sub-millisecond processing times, satisfying Objective 2 (Real-Time Efficiency).")
