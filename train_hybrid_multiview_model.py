import numpy as np
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import Lambda, Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# === Load Data ===
label_df = pd.read_csv("data/labels_cleaned.csv")
feature_df = pd.read_csv("outputs/eeg_features_full.csv")
spec_dir = "data/spectrograms_64x64"

label_df['filename'] = label_df['filename'].str.lower().str.strip()
feature_df['Filename'] = feature_df['Filename'].str.lower().str.strip()

df = pd.merge(feature_df, label_df, left_on='Filename', right_on='filename')

non_feature_cols = ['Filename', 'filename', 'label']
X_feat = df.drop(columns=non_feature_cols)
X_feat = X_feat.select_dtypes(include=[np.number]).astype(np.float32).values
y = df['label'].astype(int).values
subjects = df['Filename'].values

X_spec, valid_idx = [], []
for i, subj in enumerate(subjects):
    path = os.path.join(spec_dir, subj)
    if os.path.exists(path):
        spec = np.load(path)
        if spec.shape == (64, 64):
            X_spec.append(spec)
            valid_idx.append(i)
        else:
            print(f"⚠️ Invalid spectrogram shape for: {subj} -> {spec.shape}")
    else:
        print(f"⚠️ Missing spectrogram for: {subj}")

X_feat = X_feat[valid_idx]
y = y[valid_idx]
X_spec = np.array(X_spec, dtype=np.float32)[..., np.newaxis]

print(f"✅ Loaded {len(y)} samples. EEG Features shape: {X_feat.shape}, Spectrograms shape: {X_spec.shape}")

# === Fusion Modules ===
def trainable_gating_fusion(eeg_emb, spec_emb):
    combined = concatenate([eeg_emb, spec_emb])
    gating_weights = Dense(2, activation='softmax')(combined)
    eeg_weight = Lambda(lambda x: x[:, 0:1])(gating_weights)
    spec_weight = Lambda(lambda x: x[:, 1:2])(gating_weights)
    eeg_weighted = Multiply()([eeg_emb, eeg_weight])
    spec_weighted = Multiply()([spec_emb, spec_weight])
    fused = Add()([eeg_weighted, spec_weighted])
    return fused

def advanced_concat_fusion(eeg_emb, spec_emb):
    concat_emb = concatenate([eeg_emb, spec_emb])
    x = Dense(128, activation='relu')(concat_emb)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    return x

# === CNN Model with hybrid fusion ===
def build_cnn_hybrid_fusion(input_dim_feat):
    eeg_input = Input(shape=(input_dim_feat,), name='EEG_Features')
    x = Dense(64, activation='relu')(eeg_input)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    eeg_embedding = Dense(64, activation='relu', name='EEG_Embedding')(x)

    spec_input = Input(shape=(64, 64, 1), name='Spectrogram')
    y_ = Conv2D(16, (3, 3), activation='relu')(spec_input)
    y_ = MaxPooling2D((2, 2))(y_)
    y_ = Conv2D(32, (3, 3), activation='relu')(y_)
    y_ = MaxPooling2D((2, 2))(y_)
    y_ = Flatten()(y_)
    spec_embedding = Dense(64, activation='relu', name='Spec_Embedding')(y_)

    gating_fused = trainable_gating_fusion(eeg_embedding, spec_embedding)
    concat_fused = advanced_concat_fusion(eeg_embedding, spec_embedding)

    combined_fusion = concatenate([gating_fused, concat_fused])
    z = Dense(64, activation='relu')(combined_fusion)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid', name='output')(z)

    model = Model(inputs=[eeg_input, spec_input], outputs=[output, eeg_embedding, spec_embedding])
    model.compile(
        optimizer=Adam(1e-4),
        loss=['binary_crossentropy', None, None],
        metrics=['accuracy', None, None]
    )
    return model

# === Grad-CAM Utility function for the spectrogram branch ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# === Cross-validation with residual learning & explainability ===
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_list, f1_list, roc_list = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y), 1):
    print(f"\n=== Fold {fold} ===")

    model = build_cnn_hybrid_fusion(X_feat.shape[1])
    model.fit(
        [X_feat[train_idx], X_spec[train_idx]],
        [y[train_idx], np.zeros((len(train_idx), 64)), np.zeros((len(train_idx), 64))],
        validation_data=(
            [X_feat[val_idx], X_spec[val_idx]],
            [y[val_idx], np.zeros((len(val_idx), 64)), np.zeros((len(val_idx), 64))]
        ),
        epochs=30,
        batch_size=8,
        verbose=1
    )

    preds_train, eeg_emb_train, spec_emb_train = model.predict([X_feat[train_idx], X_spec[train_idx]])
    preds_val, eeg_emb_val, spec_emb_val = model.predict([X_feat[val_idx], X_spec[val_idx]])

    residuals_train = y[train_idx] - preds_train.flatten()

    X_stack_train = np.hstack([eeg_emb_train, spec_emb_train, X_feat[train_idx]])
    X_stack_val = np.hstack([eeg_emb_val, spec_emb_val, X_feat[val_idx]])

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_stack_train, residuals_train)

    residuals_pred_val = rf.predict(X_stack_val)
    final_pred_val = preds_val.flatten() + residuals_pred_val
    final_pred_val = np.clip(final_pred_val, 0, 1)
    final_bin_pred = (final_pred_val > 0.5).astype(int)

    acc = accuracy_score(y[val_idx], final_bin_pred)
    f1 = f1_score(y[val_idx], final_bin_pred)
    roc = roc_auc_score(y[val_idx], final_pred_val)
    print(f"Fold {fold} - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")

    # ----- Grad-CAM Visualization for spectrogram -----
    # Identify last conv layer name automatically:
    last_conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]
    last_conv_layer = last_conv_layers[-1]  # take last Conv2D layer

    # Pick a few samples from validation fold for explanation
    # Here, we visualize the first validation sample only for demonstration
    sample_idx = 0
    spec_img = X_spec[val_idx][sample_idx:sample_idx+1]
    eeg_sample = X_feat[val_idx][sample_idx:sample_idx+1]

    heatmap = make_gradcam_heatmap([eeg_sample, spec_img], model, last_conv_layer)

    # Plot Grad-CAM heatmap overlay on spectrogram
    plt.figure(figsize=(6, 5))
    plt.title(f"Fold {fold} Grad-CAM on Spectrogram Sample {sample_idx}")
    plt.imshow(spec_img[0,:,:,0], cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.show()

    # ----- SHAP Explanation for RF Residual Model -----
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_stack_val)

    # Summary plot (can save or display)
    print(f"SHAP summary plot for Fold {fold} residual correction model features:")
    shap.summary_plot(shap_values, X_stack_val, show=True)

    acc_list.append(acc)
    f1_list.append(f1)
    roc_list.append(roc)

print("\n=== Cross-Validation Summary ===")
print(f"Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"F1 Score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print(f"ROC AUC: {np.mean(roc_list):.4f} ± {np.std(roc_list):.4f}")
