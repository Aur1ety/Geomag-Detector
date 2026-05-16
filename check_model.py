import torch
import joblib
import os

model_path = r"C:\Users\ponpo\Documents\Geomag Detector\patchtransformer_mag_v1.pth"
scaler_path = r"C:\Users\ponpo\Documents\Geomag Detector\scaler_mag.pkl"

print(f"--- Identity Check ---")

# 1. Check Model Structure
state_dict = torch.load(model_path, map_location='cpu')
# We look for the first weight matrix to find the input dimension
first_weight_key = [k for k in state_dict.keys() if 'weight' in k][0]
input_dim = state_dict[first_weight_key].shape[1]

print(f"Model Input Dimensions: {input_dim}")

# 2. Check Scaler
scaler = joblib.load(scaler_path)
print(f"Scaler Features: {scaler.n_features_in_}")

if input_dim == 9:
    print("\n✅ VERIFIED: This is the 9-feature 'Smarter' Brain.")
else:
    print("\n❌ ALERT: This is still the old 8-feature model!")