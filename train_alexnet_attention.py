import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import optuna

SPECTROGRAM_DIR = "data/spectrograms_227x227_rgb"
LABEL_CSV = "data/labels_cleaned.csv"

# --- Attention Module ---
def channel_attention(inputs):
    channel_avg = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    channel_avg = tf.keras.layers.Dense(inputs.shape[-1] // 8, activation='relu')(channel_avg)
    channel_avg = tf.keras.layers.Dense(inputs.shape[-1], activation='sigmoid')(channel_avg)
    channel_avg = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(channel_avg)
    return tf.keras.layers.Multiply()([inputs, channel_avg])

# --- Attention-InceptionV3 ---
def build_inception_with_attention():
    inp = Input(shape=(227, 227, 3))
    base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_tensor=inp)
    for l in base.layers[:200]:
        l.trainable = False
    for l in base.layers[200:]:
        l.trainable = True
    x = base.output
    x = channel_attention(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Feature extractor ---
def get_feature_extractor():
    base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape=(227, 227, 3))
    return Model(inputs=base.input, outputs=base.output)

# --- Optimal threshold ---
def optimal_f1_threshold(y_true, prob_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_true, prob_pred)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1s)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

# --- Load data ---
label_df = pd.read_csv(LABEL_CSV)
label_df['filename'] = label_df['filename'].str.strip().str.lower()
X, y = [], []

for _, row in label_df.iterrows():
    fname = row['filename']
    label = int(row['label'])
    path = os.path.join(SPECTROGRAM_DIR, fname)
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape == (227, 227, 3):
            X.append(arr)
            y.append(label)

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)
print(f"✅ Loaded {len(X)} spectrograms. Shape: {X.shape}")

# --- Optuna objective for SVM ---
def optuna_svm_objective(trial, X_train_feats, y_train):
    C = trial.suggest_loguniform('C', 0.1, 10)
    gamma = trial.suggest_loguniform('gamma', 0.001, 1)
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train_feats, y_train)
    prob = clf.predict_proba(X_train_feats)[:, 1]
    f1 = f1_score(y_train, (prob > 0.5).astype(int))
    return f1

# --- Cross-validation ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, f1s, rocs = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔁 Fold {fold}")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Train CNN
    model = build_inception_with_attention()
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=8, callbacks=[stop], verbose=0)
    os.makedirs("models", exist_ok=True)
    model.save(f"models/inceptionv3_attention_fold{fold}.keras")

    # CNN predictions
    y_img_prob = model.predict(X_val).flatten()

    # Feature extraction
    feat_extractor = get_feature_extractor()
    train_feats = feat_extractor.predict(X_train)
    val_feats = feat_extractor.predict(X_val)

    # Optimize SVM with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optuna_svm_objective(trial, train_feats, y_train), n_trials=15)
    best_params = study.best_params
    print(f"  Best SVM params: {best_params}")

    svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', probability=True, random_state=42)
    svm.fit(train_feats, y_train)
    y_svm_prob = svm.predict_proba(val_feats)[:, 1]

    # Stacking: CNN + SVM
    meta_X = np.vstack([y_img_prob, y_svm_prob]).T
    meta_clf = LogisticRegression(solver='liblinear')
    meta_clf.fit(meta_X, y_val)
    meta_prob = meta_clf.predict_proba(meta_X)[:, 1]

    best_thresh = optimal_f1_threshold(y_val, meta_prob)
    y_pred = (meta_prob > best_thresh).astype(int)

    accs.append(accuracy_score(y_val, y_pred))
    f1s.append(f1_score(y_val, y_pred))
    rocs.append(roc_auc_score(y_val, meta_prob))
    print(f"  Best Threshold: {best_thresh:.3f}")
    print(f"✅ Fold {fold} → Acc: {accs[-1]:.4f} | F1: {f1s[-1]:.4f} | ROC-AUC: {rocs[-1]:.4f}")

print("\n📊 Final Attention-Based Hybrid Ensemble Metrics (SVM + CNN via Optuna):")
print(f"📈 Mean Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"📊 Mean F1 Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"📐 Mean ROC-AUC  : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
