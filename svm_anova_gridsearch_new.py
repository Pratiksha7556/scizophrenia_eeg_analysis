import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
import optuna

# === CONFIGURATION ===
feature_path = r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\eeg_features_full.csv"
k_features = 40
n_splits = 10
random_state = 42

# === LOAD DATA ===
df = pd.read_csv(feature_path)
df.dropna(inplace=True)
y = df['Label'].values
X_all = df.drop(['Filename', 'Label'], axis=1).values

# === CROSS-VALIDATION ===
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# ---- FEATURE SELECTION (MI + ANOVA) ----
mi_selector = SelectKBest(score_func=mutual_info_classif, k=80)
X_mi = mi_selector.fit_transform(X_all, y)
anova_selector = SelectKBest(score_func=f_classif, k=k_features)
X_fs = anova_selector.fit_transform(X_mi, y)

# === OPTUNA OBJECTIVE FUNCTION ===
def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    C = trial.suggest_loguniform('C', 0.01, 100)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    degree = trial.suggest_int('degree', 2, 5)  # only for poly kernel
    coef0 = trial.suggest_uniform('coef0', 0.0, 1.0)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            probability=True,
            class_weight='balanced',
            random_state=random_state
        ))
    ])

    scores = cross_val_score(pipe, X_fs, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    return np.mean(scores)

# === RUN OPTUNA STUDY ===
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)

print("\n✅ Best hyperparameters found by Optuna:")
print(study.best_params)

# === TRAIN FINAL MODEL WITH BEST PARAMETERS ===
best_params = study.best_params
best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel=best_params['kernel'],
        C=best_params['C'],
        gamma=best_params['gamma'],
        degree=best_params['degree'],
        coef0=best_params['coef0'],
        probability=True,
        class_weight='balanced',
        random_state=random_state
    ))
])

# ---- Evaluate with StratifiedKFold ----
results = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X_fs, y), 1):
    X_train, X_test = X_fs[train_idx], X_fs[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n📁 Fold {fold}")
    print(f"   Accuracy   : {acc:.4f}")
    print(f"   F1 Score   : {f1:.4f}")
    print(f"   ROC-AUC    : {roc_auc:.4f}")
    print(f"   MCC        : {mcc:.4f}")

    results.append((acc, f1, roc_auc, mcc))

# === FINAL SUMMARY ===
results = np.array(results)
print("\n📊 Final Cross-Validation Metrics (SVM + Optuna):")
print(f"   Mean Accuracy : {np.mean(results[:,0]):.4f} ± {np.std(results[:,0]):.4f}")
print(f"   Mean F1 Score : {np.mean(results[:,1]):.4f} ± {np.std(results[:,1]):.4f}")
print(f"   Mean ROC-AUC  : {np.mean(results[:,2]):.4f} ± {np.std(results[:,2]):.4f}")
print(f"   Mean MCC      : {np.mean(results[:,3]):.4f} ± {np.std(results[:,3]):.4f}")
