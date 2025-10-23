# explainability_shap_stacking.py
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

# ===============================================
# STEP 1: Load or generate data
# ===============================================
# Replace with your actual feature set and labels
X_final = np.random.rand(200, 40)   # 200 samples, 40 features
y = np.random.randint(0, 2, 200)    # binary labels

# ===============================================
# STEP 2: Train stacking ensemble
# ===============================================
meta_features = []
meta_labels = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X_final, y):
    X_train, X_test = X_final[train_idx], X_final[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Base classifiers
    svm = SVC(kernel='rbf', C=2, gamma=0.01, probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=130, max_depth=7,
                                class_weight='balanced_subsample', random_state=42)
    etc = ExtraTreesClassifier(n_estimators=150, max_depth=7,
                               bootstrap=True, class_weight='balanced_subsample', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(36, 20), max_iter=550,
                        alpha=5e-4, random_state=42)
    ridge = RidgeClassifier(alpha=0.3)
    lr = LogisticRegression(penalty='elasticnet', solver='saga',
                            l1_ratio=0.45, C=1.2, max_iter=200, random_state=42)

    for clf in [svm, rf, etc, mlp, ridge, lr]:
        clf.fit(X_train, y_train)

    ridge_scaled = (ridge.decision_function(X_test) - ridge.decision_function(X_test).min()) / \
                   (np.ptp(ridge.decision_function(X_test)) + 1e-8)

    meta_X = np.vstack([
        svm.predict_proba(X_test)[:, 1],
        rf.predict_proba(X_test)[:, 1],
        etc.predict_proba(X_test)[:, 1],
        mlp.predict_proba(X_test)[:, 1],
        ridge_scaled,
        lr.predict_proba(X_test)[:, 1]
    ]).T

    meta_features.append(meta_X)
    meta_labels.append(y_test)

meta_features = np.vstack(meta_features)
meta_labels = np.concatenate(meta_labels)

final_meta = RidgeClassifier(alpha=0.15)
final_meta.fit(meta_features, meta_labels)

# ===============================================
# STEP 3: Confusion Matrix
# ===============================================
X_train, X_test, y_train, y_test = train_test_split(meta_features, meta_labels,
                                                    test_size=0.2, stratify=meta_labels,
                                                    random_state=42)
final_meta.fit(X_train, y_train)
y_pred = final_meta.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, final_meta.decision_function(X_test))

print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Control", "Schizophrenia"],
            yticklabels=["Control", "Schizophrenia"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Stacking Ensemble")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("✅ Confusion matrix saved as confusion_matrix.png")

# ===============================================
# STEP 4: SHAP Explainability (auto-detect explainer)
# ===============================================
explainer = shap.Explainer(final_meta, meta_features)
shap_values = explainer(meta_features)

meta_feature_names = ["SVM", "RandomForest", "ExtraTrees", "MLP", "Ridge", "LogReg"]

# SHAP summary plot
shap.summary_plot(shap_values.values, meta_features,
                  feature_names=meta_feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

# SHAP bar plot
shap.summary_plot(shap_values.values, meta_features,
                  feature_names=meta_feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300)
plt.close()

print("✅ SHAP plots saved: shap_summary.png, shap_feature_importance.png")
