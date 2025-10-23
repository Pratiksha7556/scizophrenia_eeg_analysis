import os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === CONFIG ===
X_path = r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\X_pca_50.npy"
y_path = r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\y_pca.npy"
pca_components = 50
n_splits = 5
random_state = 42

np.random.seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

# === LOAD PCA DATA ===
X_pca = np.load(X_path)
y = np.load(y_path)

# === EXTRACT SUBJECT IDs from filenames ===
# Assumes filenames are in same order as y and X in eeg_features_full.csv
import pandas as pd
df = pd.read_csv(r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\eeg_features_full.csv")
subjects = df['Filename'].apply(lambda x: x.replace('.npy', '')).values

assert X_pca.shape[0] == y.shape[0] == len(subjects) == 81, "Shape mismatch in loaded data"
assert X_pca.shape[1] == pca_components, "PCA components mismatch"

# === CROSS-VALIDATION ===
gkf = GroupKFold(n_splits=n_splits)
accs, f1s, aucs = [], [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_pca, y, groups=subjects), 1):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # === Build Model ===
    model = Sequential([
        Dense(128, activation='relu', input_shape=(pca_components,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2,
              callbacks=[es, rlrop], verbose=0)

    # === Evaluate ===
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n📁 Fold {fold}")
    print(f"   Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print("   Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    accs.append(acc)
    f1s.append(f1)
    aucs.append(auc)

# === Final Results ===
print("\n📊 Final Cross-Validation Performance:")
print(f"   Mean Accuracy: {np.mean(accs):.4f}")
print(f"   Mean F1 Score: {np.mean(f1s):.4f}")
print(f"   Mean ROC-AUC : {np.mean(aucs):.4f}")
print(f"   Std Accuracy : {np.std(accs):.4f}")
