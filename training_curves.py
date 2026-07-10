
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import RMSprop
from keras import regularizers
import tensorflow as tf

# Load data
data = pd.read_csv('/content/drive/MyDrive/Term_Project/Code/PReNet/data/annthyroid_21feat_normalised.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

def regression_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)

def build_prenet(input_dim):
    x_input = Input(shape=(input_dim,))
    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01))(x_input)
    base_network = Model(x_input, intermediate)
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    merged = concatenate([processed_a, processed_b])
    score = Dense(1, activation='linear')(merged)
    model = Model([input_a, input_b], score)
    model.compile(loss=regression_loss, optimizer=RMSprop(clipnorm=1.))
    return model

def make_pairs(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, seed):
    uu, au, aa = 0, 4, 8
    rng = np.random.RandomState(seed)
    all_s1, all_s2, all_labels = [], [], []
    for _ in range(nb_batch):
        dim = x_train.shape[1]
        pairs1 = np.empty((batch_size, dim))
        pairs2 = np.empty((batch_size, dim))
        labels = []
        n_in = len(inlier_indices)
        n_out = len(outlier_indices)
        block = batch_size // 4
        sid = rng.choice(n_in, block*4, replace=False)
        pairs1[0:block*2] = x_train[inlier_indices[sid[0:block*2]]]
        pairs2[0:block*2] = x_train[inlier_indices[sid[block*2:block*4]]]
        labels += block*2*[uu]
        sid = rng.choice(n_in, block, replace=False)
        pairs1[block*2:block*3] = x_train[inlier_indices[sid]]
        sid = rng.choice(n_out, block)
        pairs2[block*2:block*3] = x_train[outlier_indices[sid]]
        labels += block*[au]
        for i in range(block*3, batch_size):
            sid = rng.choice(n_out, 2, replace=False)
            pairs1[i] = x_train[outlier_indices[sid[0]]]
            pairs2[i] = x_train[outlier_indices[sid[1]]]
            labels += [aa]
        all_s1.append(pairs1)
        all_s2.append(pairs2)
        all_labels.append(np.array(labels, dtype=float))
    return (np.concatenate(all_s1).astype(np.float32),
            np.concatenate(all_s2).astype(np.float32),
            np.concatenate(all_labels).astype(np.float32))

# Use seed 0 for the plot
np.random.seed(0)
tf.random.set_seed(0)
known_outliers = 30
epochs = 50
nb_batch = 20
batch_size = 512

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

outlier_indices = np.where(y_train == 1)[0]
rng = np.random.RandomState(0)
if len(outlier_indices) > known_outliers:
    remove_idx = rng.choice(outlier_indices,
                            len(outlier_indices) - known_outliers,
                            replace=False)
    x_train = np.delete(x_train, remove_idx, axis=0)
    y_train = np.delete(y_train, remove_idx, axis=0)

outlier_indices = np.where(y_train == 1)[0]
inlier_indices  = np.where(y_train == 0)[0]

model = build_prenet(x_train.shape[1])

loss_history = []
print("Training for plotting...")
for epoch in range(epochs):
    s1, s2, labels = make_pairs(x_train, outlier_indices,
                                inlier_indices, batch_size, nb_batch, epoch)
    hist = model.fit([s1, s2], labels,
                     batch_size=batch_size,
                     epochs=1, verbose=0)
    loss_history.append(hist.history['loss'][0])
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | Loss: {loss_history[-1]:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, epochs+1), loss_history, color='#2E4057', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss (MAE)', fontsize=12)
ax.set_title('PReNet Training Loss on Annthyroid (Seed 0)', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Term_Project/results/training_curve.png', dpi=150)
plt.show()
print("Plot saved.")