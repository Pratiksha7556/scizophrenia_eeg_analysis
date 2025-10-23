import numpy as np
import pandas as pd
import optuna
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
from tensorflow.keras.optimizers import Adam

# ========= Config section =========
feature_path = r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\eeg_features_full.csv"
random_state = 42
n_splits = 10
np.random.seed(random_state)

# ========= Data Loading =========
df = pd.read_csv(feature_path)
df.dropna(inplace=True)
y = df['Label'].values
subjects = df['Filename'].apply(lambda x: x.replace('.npy', '')).values
X = df.drop(columns=['Filename', 'Label']).values

# ========= Define ML Base Models =========
base_models = [
    ('LR', LogisticRegression(max_iter=1000, random_state=random_state)),
    ('DT', DecisionTreeClassifier(random_state=random_state)),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ('SVM', SVC(probability=True, random_state=random_state)),
    ('KNN', KNeighborsClassifier()),
    ('NB', GaussianNB())
]

# ========= CNN Maker =========
def make_cnn(input_shape, filters=32, kernel_size=3, dropout=0.5, lr=0.001):
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# ========= Optuna tuning for ML models =========
def tune_ml_model(model_name, X_train, y_train, subjects_train, n_trials=30):
    def objective(trial):
        if model_name == 'RF':
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        elif model_name == 'DT':
            max_depth = trial.suggest_int("max_depth", 1, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
        elif model_name == 'KNN':
            n_neighbors = trial.suggest_int("n_neighbors", 1, 15)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_name == 'SVM':
            C = trial.suggest_float("C", 0.1, 10.0, log=True)
            gamma = trial.suggest_float("gamma", 0.001, 0.1, log=True)
            clf = SVC(C=C, gamma=gamma, probability=True, random_state=random_state)
        else:
            return 0.5

        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in inner_cv.split(X_train, y_train, groups=subjects_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            clf.fit(X_tr, y_tr)
            y_pred_prob = clf.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred_prob))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# ========= Optuna tuning for CNN =========
def tune_cnn_model(X_train, y_train, n_trials=20):
    def objective(trial):
        filters = trial.suggest_int('filters', 16, 64)
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        dropout = trial.suggest_float('dropout', 0.3, 0.6)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

        X_tr, X_val = X_train[:int(0.8*len(X_train))], X_train[int(0.8*len(X_train)):]
        y_tr, y_val = y_train[:int(0.8*len(y_train))], y_train[int(0.8*len(y_train)):]

        model = make_cnn(input_shape=(X_train.shape[1],1), filters=filters,
                         kernel_size=kernel_size, dropout=dropout, lr=lr)
        history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                            epochs=30, batch_size=batch_size,
                            verbose=0)
        return max(history.history['val_accuracy'])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# ========= Stacking Ensemble =========
meta_features, meta_labels, meta_subjects = [], [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=subjects), 1):
    print(f"\n=== FOLD {fold} ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    subjects_train = subjects[train_idx]
    subjects_test = subjects[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fold_base_preds = []

    # Tune and fit ML models
    for name, _ in base_models:
        if name in ['RF','DT','SVM','KNN']:
            print(f"Tuning {name}...")
            best_params = tune_ml_model(name, X_train_scaled, y_train, subjects_train)
            if name == 'RF':
                model = RandomForestClassifier(**best_params, random_state=random_state)
            elif name == 'DT':
                model = DecisionTreeClassifier(**best_params, random_state=random_state)
            elif name == 'SVM':
                model = SVC(**best_params, probability=True, random_state=random_state)
            elif name == 'KNN':
                model = KNeighborsClassifier(**best_params)
        else:
            model = dict(base_models)[name]

        model.fit(X_train_scaled, y_train)
        pred_prob = model.predict_proba(X_test_scaled)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_test_scaled)
        fold_base_preds.append(pred_prob.reshape(-1,1))
        pred = (pred_prob>0.5).astype(int)
        print(f"  {name} | Acc: {accuracy_score(y_test,pred):.3f} F1: {f1_score(y_test,pred):.3f} ROC-AUC: {roc_auc_score(y_test,pred_prob):.3f}")

    # Tune and fit CNN
    print("Tuning CNN...")
    best_cnn = tune_cnn_model(X_train_scaled[...,np.newaxis], y_train)
    cnn = make_cnn(input_shape=(X_train_scaled.shape[1],1),
                   filters=best_cnn['filters'],
                   kernel_size=best_cnn['kernel_size'],
                   dropout=best_cnn['dropout'],
                   lr=best_cnn['lr'])
    batch_size = best_cnn['batch_size']
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    cnn.fit(X_train_scaled[...,np.newaxis], y_train,
            validation_split=0.2, epochs=50, batch_size=batch_size,
            callbacks=[early_stop, reduce_lr], verbose=0)
    cnn_pred_prob = cnn.predict(X_test_scaled[...,np.newaxis]).flatten()
    fold_base_preds.append(cnn_pred_prob.reshape(-1,1))
    cnn_pred = (cnn_pred_prob>0.5).astype(int)
    print(f"  CNN | Acc: {accuracy_score(y_test,cnn_pred):.3f} F1: {f1_score(y_test,cnn_pred):.3f} ROC-AUC: {roc_auc_score(y_test,cnn_pred_prob):.3f}")

    # Stack
    meta_features.append(np.hstack(fold_base_preds))
    meta_labels.append(y_test)
    meta_subjects.append(subjects_test)

# Prepare meta dataset
meta_features = np.vstack(meta_features)
meta_labels = np.concatenate(meta_labels)
meta_subjects = np.concatenate(meta_subjects)

# Meta-learner
meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
accs, f1s, aucs = [], [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(meta_features, meta_labels, groups=meta_subjects),1):
    X_train_meta, X_test_meta = meta_features[train_idx], meta_features[test_idx]
    y_train_meta, y_test_meta = meta_labels[train_idx], meta_labels[test_idx]
    meta_model.fit(X_train_meta, y_train_meta)
    y_pred_prob = meta_model.predict_proba(X_test_meta)[:,1]
    y_pred = (y_pred_prob>0.5).astype(int)
    accs.append(accuracy_score(y_test_meta,y_pred))
    f1s.append(f1_score(y_test_meta,y_pred))
    aucs.append(roc_auc_score(y_test_meta,y_pred_prob))
    print(f"Meta Fold {fold} | Acc: {accs[-1]:.3f} F1: {f1s[-1]:.3f} ROC-AUC: {aucs[-1]:.3f}")

print(f"\n=== Final Model ===")
print(f"Mean Acc: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Mean F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"Mean ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
