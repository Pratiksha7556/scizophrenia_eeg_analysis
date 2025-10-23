import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

# ========= Load Data =========
df = pd.read_csv(r"C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\eeg_features_full.csv")
df.dropna(inplace=True)
y = df['Label'].values
X = df.drop(columns=['Filename', 'Label']).values
feature_names = df.drop(columns=['Filename', 'Label']).columns

# ========= Preprocess =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========= Define Base Models =========
base_models = [
    ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ("DT", DecisionTreeClassifier(random_state=42)),
    ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVM", SVC(probability=True, random_state=42)),
    ("KNN", KNeighborsClassifier()),
    ("NB", GaussianNB())
]

# ========= CNN Model =========
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

X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]
cnn = make_cnn((X_train.shape[1], 1))
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

cnn.fit(
    X_train_cnn, y_train, epochs=30, batch_size=8, validation_split=0.2,
    callbacks=[early_stop, reduce_lr], verbose=0
)

# ========= Collect Base Predictions =========
train_preds, test_preds = [], []
for name, model in base_models:
    model.fit(X_train, y_train)
    train_preds.append(model.predict_proba(X_train)[:, 1].reshape(-1, 1))
    test_preds.append(model.predict_proba(X_test)[:, 1].reshape(-1, 1))

# CNN predictions
train_preds.append(cnn.predict(X_train_cnn).reshape(-1, 1))
test_preds.append(cnn.predict(X_test_cnn).reshape(-1, 1))

# Stack predictions
X_train_meta = np.hstack(train_preds)
X_test_meta = np.hstack(test_preds)
meta_feature_names = [name for name, _ in base_models] + ["CNN"]

# ========= Meta Learner =========
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(X_train_meta, y_train)

y_pred_prob = meta_model.predict_proba(X_test_meta)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("Fusion ML + 1D-CNN Performance:")
print("Accuracy:", acc)
print("F1 Score:", f1)
print("ROC-AUC:", auc)

# ========= SHAP Explainability =========
explainer_meta = shap.Explainer(meta_model, X_train_meta, feature_names=meta_feature_names)
shap_values_meta = explainer_meta(X_test_meta)

# Plot SHAP summary (bar)
shap.summary_plot(shap_values_meta, X_test_meta, feature_names=meta_feature_names,
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("fusion_meta_shap_bar.png", dpi=300)
plt.clf()

# Plot SHAP summary (beeswarm)
shap.summary_plot(shap_values_meta, X_test_meta, feature_names=meta_feature_names, show=False)
plt.tight_layout()
plt.savefig("fusion_meta_shap_beeswarm.png", dpi=300)
plt.clf()

# ========= Extra Plots =========
# 1. Performance bar plot
metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
scores = [acc, f1, auc]

plt.figure(figsize=(6,4))
bars = plt.bar(metrics, scores, color=["skyblue", "lightgreen", "salmon"])
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
             f"{score:.2f}", ha='center', va='bottom')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Fusion ML + 1D-CNN Performance")
plt.tight_layout()
plt.savefig("fusion_model_performance.png", dpi=300)
plt.clf()

# 2. SHAP mean importance bar (quick view)
mean_shap = np.abs(shap_values_meta.values).mean(axis=0)
plt.figure(figsize=(6,4))
bars = plt.bar(meta_feature_names, mean_shap, color="orange")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean |SHAP value|")
plt.title("Base Model Contribution to Meta-Learner")
plt.tight_layout()
plt.savefig("fusion_meta_base_contribution.png", dpi=300)
plt.clf()
