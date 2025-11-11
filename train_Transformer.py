import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import os

# === Load the labeled dataset ===
file_path = "C:/Users/21653/Desktop/python/theseTransformer/datafinaleTransformer.xlsx"
df = pd.read_excel(file_path)

# === Mapping des classes ===
class_mapping = {
    1: "Defect 1 (theta1 stuck), 0.1, motor 1 stuck",
    2: "Defect 1 (theta1 stuck), 0.2, motor 1 stuck",
    3: "Defect 1 (theta1 stuck), 0.3, motor 1 stuck",
    4: "Defect 2 (theta2 stuck), 0.1, motor 2 stuck",
    5: "Defect 2 (theta2 stuck), 0.2, motor 2 stuck",
    6: "Defect 2 (theta2 stuck), 0.3, motor 2 stuck",
    7: "Defect 3 (theta3 stuck), 0.1, motor 3 stuck",
    8: "Defect 3 (theta3 stuck), 0.2, motor 3 stuck",
    9: "Defect 3 (theta3 stuck), 0.3, motor 3 stuck",
    10: "Defect 4 (theta4 stuck), 0.1, motor 4 stuck",
    11: "Defect 4 (theta4 stuck), 0.2, motor 4 stuck",
    12: "Defect 4 (theta4 stuck), 0.3, motor 4 stuck",
    13: "Defect 5 (theta5 stuck), 0.1, motor 5 stuck",
    14: "Defect 5 (theta5 stuck), 0.2, motor 5 stuck",
    15: "Defect 5 (theta5 stuck), 0.3, motor 5 stuck",
    16: "Defect 6 (theta6 stuck), 0.1, motor 6 stuck",
    17: "Defect 6 (theta6 stuck), 0.2, motor 6 stuck",
    18: "Defect 6 (theta6 stuck), 0.3, motor 6 stuck",
    19: "Normal State"
}

# === Construction des séquences ===
sequence_length = 5
features = df.columns[:-1]
X_raw = df[features].values
y_raw = df["Class"].values

X, y = [], []
for i in range(0, len(X_raw) - sequence_length + 1):
    x_seq = X_raw[i:i + sequence_length]
    y_seq = y_raw[i:i + sequence_length]
    counter = Counter(y_seq)
    most_common_class, count = counter.most_common(1)[0]
    if count >= 4:
        X.append(x_seq)
        y.append(most_common_class)

X = np.array(X)
y = np.array(y)

print(f"Nombre de séquences valides : {len(X)}")
if len(X) < 2:
    raise ValueError("Pas assez de séquences valides pour entraîner le modèle.")

# === Encodage des labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# === Split & normalisation ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X.shape[2])).reshape(-1, sequence_length, X.shape[2])
X_test = scaler.transform(X_test.reshape(-1, X.shape[2])).reshape(-1, sequence_length, X.shape[2])

# === One-hot encoding ===
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# === Positional Encoding Layer ===
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# === Transformer Encoder Block ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = tf.keras.layers.Dense(ff_dim, activation="relu")(res)
    x = tf.keras.layers.Dense(inputs.shape[-1])(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# === Transformer Model ===
input_shape = (sequence_length, X.shape[2])
inputs = tf.keras.Input(shape=input_shape)
x = PositionalEncoding(sequence_length, X.shape[2])(inputs)
x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === Training ===
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=100, batch_size=32)

# === Prediction ===
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# === Evaluation ===
accuracy = accuracy_score(y_test, y_pred)
print(f"Transformer Model Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=[class_mapping[label_encoder.inverse_transform([i])[0]] for i in range(num_classes)],
    columns=[class_mapping[label_encoder.inverse_transform([i])[0]] for i in range(num_classes)]
)

y_test_names = [class_mapping[label] for label in label_encoder.inverse_transform(y_test)]
y_pred_names = [class_mapping[label] for label in label_encoder.inverse_transform(y_pred)]
class_report = classification_report(y_test_names, y_pred_names, output_dict=True, zero_division=0)
class_report_df = pd.DataFrame(class_report).transpose()

# === Save results ===
save_folder = "C:/Users/21653/Desktop/python/theseTransformer/final"
os.makedirs(save_folder, exist_ok=True)
conf_matrix_df.to_excel(os.path.join(save_folder, "transformer_confusion_matrix.xlsx"))
class_report_df.to_excel(os.path.join(save_folder, "transformer_classification_report.xlsx"))

# === Accuracy plot ===
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy (Transformer)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_folder, "transformer_accuracy_plot.png"))
plt.show()

# === Confusion matrix plot ===
plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Transformer)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "transformer_confusion_matrix.png"))
plt.show()

# === Save model ===
model.save(os.path.join(save_folder, "transformer_model.h5"))
