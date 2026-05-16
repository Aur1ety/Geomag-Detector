import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from scipy.stats import norm

# 1. Path Management
sys.path.insert(0, os.getcwd())

from model_factory import PatchTransformer
from mag_pipeline import load_mag_directory
from feature_engineer import build_mag_features

# --- CONFIGURATION ---
MODEL_PATH = "patchtransformer_mag_v1.pth"
SCALER_PATH = "scaler_mag.pkl"
THRESHOLD = 0.61  # Optimal F1 threshold from training

TEST_DIRS = [
    r"C:\Users\ponpo\Documents\Geomag Detector\Data\blind test\Apr 12_26 - Apr 14_26",
    r"C:\Users\ponpo\Documents\Geomag Detector\Data\blind test\Oct 9_24 - Oct 12_24"
]

def apply_viterbi_filter(probs):
    """
    Final Production Version:
    1. Extreme Inertia HMM (Transition Penalty: 10^-7)
    2. Minimum Duration Filter (15-minute Debounce)
    """
    n = len(probs)
    if n == 0: return np.array([])

    # --- 1. THE "SUPER-STUBBORN" HMM ---
    # Quiet -> CME now costs 10^-7. 
    # This requires about 10-15 mins of sustained high prob to overcome.
    T = np.array([
        [1 - 1e-7, 1e-7], 
        [1e-4, 1 - 1e-4]
    ])
    
    means = [0.15, 0.95] # Move CME mean higher to ignore mid-level noise
    stds = [0.20, 0.10]  # Narrower CME variance makes it harder to "guess" a CME
    
    viterbi = np.zeros((2, n))
    backpointer = np.zeros((2, n), dtype=int)
    
    viterbi[0, 0] = np.log(1.0) + norm.logpdf(probs[0], means[0], stds[0])
    viterbi[1, 0] = np.log(1e-15) + norm.logpdf(probs[0], means[1], stds[1])
    
    for t in range(1, n):
        for s in range(2):
            emission_log_prob = norm.logpdf(probs[t], means[s], stds[s])
            probs_from_prev = [viterbi[prev_s, t-1] + np.log(T[prev_s, s]) for prev_s in range(2)]
            viterbi[s, t] = emission_log_prob + max(probs_from_prev)
            backpointer[s, t] = np.argmax(probs_from_prev)
            
    best_path = np.zeros(n, dtype=int)
    best_path[n-1] = np.argmax(viterbi[:, n-1])
    for t in range(n-2, -1, -1):
        best_path[t] = backpointer[best_path[t+1], t+1]
        
    # --- 2. THE DEBOUNCE LOGIC (The "Noise Killer") ---
    # If a CME detection lasts less than 15 minutes, it's a false positive.
    # We use a simple contiguous-block check.
    path_series = pd.Series(best_path)
    # Group contiguous identical values
    groups = (path_series != path_series.shift()).cumsum()
    
    for _, group in path_series.groupby(groups):
        if group.iloc[0] == 1 and len(group) < 15: # 15 minute threshold
            best_path[group.index] = 0
            
    return best_path

def run_inference():
    print("🧠 Initializing Hybrid Transformer-HMM (Pure Python) Brain...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find {MODEL_PATH}.")
        return

    # Load Model & Scaler
    model = PatchTransformer(input_dim=9)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    
    for test_dir in TEST_DIRS:
        folder_name = os.path.basename(test_dir)
        print(f"\n🔍 Analyzing Mission Data: {folder_name}")
        
        # 3. Data Processing
        raw_df = load_mag_directory(test_dir)
        features = build_mag_features(raw_df)
        
        feature_cols = [
            'bz', 'b_mag', 'clock_angle', 'dbz_dt', 'b_rotation', 
            'bz_smoothed', 'bz_persistence', 'b_elevation',
            'high_b_mag_rotation'
        ]
        
        X_scaled = scaler.transform(features[feature_cols])
        
        # 4. Sliding Window Inference
        raw_probs = []
        times = []
        win_size = 128
        
        print(f"🏃 Running Transformer inference...")
        with torch.no_grad():
            for i in range(win_size, len(X_scaled)):
                seq = X_scaled[i-win_size:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                output = model(seq_tensor)
                prob = torch.sigmoid(output).item()
                raw_probs.append(prob)
                times.append(features.index[i])

        # 5. Apply Viterbi Filter
        print("🛠 Refining detections with Viterbi Algorithm...")
        clean_states = apply_viterbi_filter(raw_probs)

        # 6. Visualization
        plt.figure(figsize=(15, 7))
        
        # Plot 1: Raw Probability (Noisy Crimson)
        plt.plot(times, raw_probs, color='crimson', lw=1, alpha=0.3, label='Transformer Raw Prob')
        
        # Plot 2: HMM Detection (Solid Orange Block)
        plt.fill_between(times, 0, clean_states, color='orange', alpha=0.35, label='CME Detected (HMM)')
        
        # Plot 3: The Step Line (Final Decision)
        plt.step(times, clean_states, color='black', lw=1.8, label='HMM Hidden State')

        # Plot 4: Reference Threshold
        plt.axhline(y=THRESHOLD, color='gray', linestyle=':', alpha=0.5, label='Inference Threshold')

        plt.title(f"Hybrid Transformer-HMM Detection Report: {folder_name}", fontsize=14)
        plt.ylabel("CME Probability / Binary State")
        plt.xlabel("Time (UTC)")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.15)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_inference()