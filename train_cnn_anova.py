import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ========= Config section =========
feature_path = r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\eeg_features_full.csv"
random_state = 42
n_splits = 5
np.random.seed(random_state)


# ========= Data Loading =========
df = pd.read_csv(feature_path)
df.dropna(inplace=True)
y = df['Label'].values
subjects = df['Filename'].apply(lambda x: x.replace('.npy', '')).values
X = df.drop(columns=['Filename', 'Label']).values


# ========= Define ML Base Models =========
models = [
    ('LR', LogisticRegression(max_iter=1000, random_state=random_state)),
    ('DT', DecisionTreeClassifier(random_state=random_state)),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ('SVM', SVC(probability=True, random_state=random_state)),
    ('KNN', KNeighborsClassifier()),
    ('NB', GaussianNB())
]


def make_cnn(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.5),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ========= Stacking Ensemble (with CNN) =========
meta_features = []  # Out-of-fold train features for meta-learner
meta_labels = []
meta_subjects = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=subjects), 1):
    print(f"\n=== FOLD {fold} ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    fold_subjects = subjects[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fold_base_preds = []

    # Classic ML models with immediate performance printing
    for name, model in models:
        model.fit(X_train_scaled, y_train)
        if hasattr(model, 'predict_proba'):
            pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            pred_prob = model.decision_function(X_test_scaled)
        pred = (pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, pred_prob)
        fold_base_preds.append(pred_prob.reshape(-1, 1))
        print(f"  {name} | Acc: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")

    # CNN model
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_test_cnn = X_test_scaled[..., np.newaxis]
    cnn = make_cnn(input_shape=(X.shape[1], 1))
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=8, validation_split=0.2,
            callbacks=[early_stop, reduce_lr], verbose=0)
    cnn_pred_prob = cnn.predict(X_test_cnn).flatten()
    cnn_pred = (cnn_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, cnn_pred)
    f1 = f1_score(y_test, cnn_pred)
    auc = roc_auc_score(y_test, cnn_pred_prob)
    fold_base_preds.append(cnn_pred_prob.reshape(-1, 1))
    print(f"  CNN | Acc: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")

    stack_pred = np.hstack(fold_base_preds)

    meta_features.append(stack_pred)
    meta_labels.append(y_test)
    meta_subjects.append(fold_subjects)


# Prepare meta dataset
meta_features = np.vstack(meta_features)
meta_labels = np.concatenate(meta_labels)
meta_subjects = np.concatenate(meta_subjects)


# Meta-learner
meta_model = LogisticRegression(max_iter=1000, random_state=random_state)

accs, f1s, aucs = [], [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(meta_features, meta_labels, groups=meta_subjects), 1):
    X_train_meta, X_test_meta = meta_features[train_idx], meta_features[test_idx]
    y_train_meta, y_test_meta = meta_labels[train_idx], meta_labels[test_idx]
    meta_model.fit(X_train_meta, y_train_meta)
    y_pred_prob = meta_model.predict_proba(X_test_meta)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test_meta, y_pred)
    f1 = f1_score(y_test_meta, y_pred)
    auc = roc_auc_score(y_test_meta, y_pred_prob)
    accs.append(acc)
    f1s.append(f1)
    aucs.append(auc)
    print(f"Meta Fold {fold} | Acc: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")


print(f"\n=== Final Model ===")
print(f"Mean Acc: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Mean F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"Mean ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
