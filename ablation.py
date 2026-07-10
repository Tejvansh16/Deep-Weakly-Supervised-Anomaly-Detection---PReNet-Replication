
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras import regularizers
import tensorflow as tf

# Load data
data = pd.read_csv('/content/drive/MyDrive/Term_Project/Code/PReNet/data/annthyroid_21feat_normalised.csv')
print("Data shape:", data.shape)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

def build_model(input_dim):
    x_input = Input(shape=(input_dim,))
    hidden = Dense(20, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01))(x_input)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(x_input, output)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipnorm=1.))
    return model

seeds = [0, 1, 2, 3, 4]
known_outliers = 30
aucs = []
aps = []

for seed in seeds:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    outlier_indices = np.where(y_train == 1)[0]
    n_outliers = len(outlier_indices)

    # Keep only K known outliers
    rng = np.random.RandomState(seed)
    if n_outliers > known_outliers:
        remove_idx = rng.choice(outlier_indices,
                                n_outliers - known_outliers,
                                replace=False)
        x_train = np.delete(x_train, remove_idx, axis=0)
        y_train = np.delete(y_train, remove_idx, axis=0)

    print(f"Seed {seed} | Train size: {x_train.shape[0]}, Outliers: {int(y_train.sum())}")

    model = build_model(x_train.shape[1])
    model.fit(x_train, y_train,
              epochs=50,
              batch_size=512,
              verbose=0)

    scores = model.predict(x_test).flatten()
    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    aucs.append(auc)
    aps.append(ap)
    print(f"Seed {seed} | AUC-ROC: {auc:.4f}, AUC-PR: {ap:.4f}")

print(f"\nAblation Mean AUC-ROC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
print(f"Ablation Mean AUC-PR:  {np.mean(aps):.4f} +/- {np.std(aps):.4f}")

# Save results
ablation = pd.DataFrame({'seed': seeds, 'auc_roc': aucs, 'auc_pr': aps})
ablation.to_csv('/content/drive/MyDrive/Term_Project/results/ablation_results.csv', index=False)
print("\nAblation results saved.")