import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    roc_auc_score, precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# CONFIG
feature_path = r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\eeg_features_full.csv"
k_features = 40
n_splits = 5
random_state = 42
np.random.seed(random_state)

# LOAD DATA
df = pd.read_csv(feature_path)
df.dropna(inplace=True)
df['subject'] = df['Filename'].str.replace('.npy', '', regex=False)
subjects = df['subject'].values
y = df['Label'].values
X_total = df.drop(columns=['Filename', 'Label', 'subject']).to_numpy()
assert X_total.shape == (81, 144), f"Expected (81, 144), got {X_total.shape}"

cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
accs, f1s, aucs = [], [], []

def optimal_f1_threshold(y_true, prob_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_true, prob_pred)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1s)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

for fold, (train_idx, test_idx) in enumerate(cv.split(X_total, y, groups=subjects), 1):
    # Per-fold scaling and selection
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_total[train_idx])
    X_test_scaled = scaler.transform(X_total[test_idx])
    selector = SelectKBest(f_classif, k=k_features)
    X_train = selector.fit_transform(X_train_scaled, y[train_idx])
    X_test = selector.transform(X_test_scaled)

    # Base learners (ensure ALL support predict_proba)
    estimators = [
        ('svm', SVC(kernel='rbf', probability=True, random_state=random_state)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)),
        ('lgbm', LGBMClassifier(random_state=random_state))
    ]
    # Meta-learner with predict_proba (LogisticRegression)
    final_estimator = LogisticRegression(max_iter=1000, random_state=random_state)

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=True,
        cv=3,
        n_jobs=-1
    )

    stacking.fit(X_train, y[train_idx])
    y_prob = stacking.predict_proba(X_test)[:, 1]
    best_thresh = optimal_f1_threshold(y[test_idx], y_prob)
    y_pred_opt = (y_prob > best_thresh).astype(int)
    
    acc = accuracy_score(y[test_idx], y_pred_opt)
    f1 = f1_score(y[test_idx], y_pred_opt)
    auc = roc_auc_score(y[test_idx], y_prob)

    print(f"\n📁 Fold {fold}")
    print(f"   Best threshold: {best_thresh:.3f}")
    print(f"   Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y[test_idx], y_pred_opt, digits=4))

    accs.append(acc)
    f1s.append(f1)
    aucs.append(auc)

print("\n📊 Final Cross-Validation Performance (Stacked Ensemble):")
print(f"   Mean Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"   Mean F1 Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"   Mean ROC-AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
