import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.signal import welch
from scipy.integrate import trapezoid

# === CONFIG ===
base_dir = r'C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\data\np_data'
output_csv = r'C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\eeg_features_human_readable.csv'
bad_subjects = {"401w1", "s43w1", "S153W1"}  # dropped due to imputation or QC failure

# === Electrode Channel Labels ===
channel_labels = ["F7", "F3", "F4", "F8",
                  "T3", "C3", "Cz", "C4",
                  "T4", "T5", "P3", "Pz",
                  "P4", "T6", "O1", "O2"]

# === EEG Feature Functions ===
def bandpower(data, sf, band, window_sec=None):
    band = np.array(band)
    low, high = band
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = None
    freqs, psd = welch(data, sf, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    return trapezoid(psd[mask], freqs[mask])

def petrosian_fd(signal):
    """Petrosian Fractal Dimension"""
    diff = np.diff(signal)
    N_delta = np.sum(diff[:-1] * diff[1:] < 0)
    n = len(signal)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

def extract_features(eeg, sf=128):
    features = {}
    for ch_idx, ch_data in enumerate(eeg):
        label = channel_labels[ch_idx]
        features[f'{label}_delta'] = bandpower(ch_data, sf, [0.5, 4])
        features[f'{label}_theta'] = bandpower(ch_data, sf, [4, 8])
        features[f'{label}_alpha'] = bandpower(ch_data, sf, [8, 12])
        features[f'{label}_beta']  = bandpower(ch_data, sf, [12, 30])
        features[f'{label}_gamma'] = bandpower(ch_data, sf, [30, 100])
        features[f'{label}_mean']  = np.mean(ch_data)
        features[f'{label}_std']   = np.std(ch_data)
        features[f'{label}_entropy'] = entropy(np.abs(np.fft.fft(ch_data)))
        features[f'{label}_pfd']   = petrosian_fd(ch_data)
    return features

# === Process Clean EEG Files Only ===
all_features = []
for label_name in ['healthy', 'schizophrenia']:
    label = 0 if label_name == 'healthy' else 1
    folder = os.path.join(base_dir, label_name)

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.npy'):
            continue
        subj_id = fname.replace('.npy', '')
        if subj_id in bad_subjects:
            print(f"🚫 Skipping bad subject: {subj_id}")
            continue

        try:
            path = os.path.join(folder, fname)
            eeg = np.load(path)
            feats = extract_features(eeg)
            feats['Filename'] = fname
            feats['Label'] = label
            all_features.append(feats)
            print(f"✅ Extracted: {fname}")
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

# === Save to CSV ===
df = pd.DataFrame(all_features)
df.to_csv(output_csv, index=False)
print(f"\n📁 Saved all features to: {output_csv}")
