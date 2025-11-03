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
file_path = "C:/Users/21653/Downloads/these/datafinalelstm.xlsx"
df = pd.read_excel(file_path)

# === Mapping des classes ===
class_mapping = {
    1: "Defect 1 (theta1 stuck), 0.1, motor 1 stuck",
    2: "Defect 1 (theta1 stuck), 0.2, motor 1 stuck",
    3: "Defect 1 (theta1 stuck), 0.3, motor 1 stuck",
    4: "Defect 3 (theta2 stuck), 0.1, motor 2 stuck",
    5: "Defect 3 (theta2 stuck), 0.2, motor 2 stuck",
    6: "Defect 3 (theta2 stuck), 0.3, motor 2 stuck",
    7: "Defect 5 (theta3 stuck), 0.1, motor 3 stuck",
    8: "Defect 5 (theta3 stuck), 0.2, motor 3 stuck",
    9: "Defect 5 (theta3 stuck), 0.3, motor 3 stuck",
    10: "Defect 7 (theta4 stuck), 0.1, motor 4 stuck",
    11: "Defect 7 (theta4 stuck), 0.2, motor 4 stuck",
    12: "Defect 7 (theta4 stuck), 0.3, motor 4 stuck",
    13: "Defect 9 (theta5 stuck), 0.1, motor 5 stuck",
    14: "Defect 9 (theta5 stuck), 0.2, motor 5 stuck",
    15: "Defect 9 (theta5 stuck), 0.3, motor 5 stuck",
    16: "Defect 11 (theta6 stuck), 0.1, motor 6 stuck",
    17: "Defect 11 (theta6 stuck), 0.2, motor 6 stuck",
    18: "Defect 11 (theta6 stuck), 0.3, motor 6 stuck",
    19: "Normal State"
}

# === Construction des séquences ===
sequence_length = 5
features = df.columns[:-1]
X_raw = df[features].values
y_raw = df["Class"].values

X = []
y = []

for i in range(0, len(X_raw) - sequence_length + 1):  # glissement sans saut
    x_seq = X_raw[i:i + sequence_length]
    y_seq = y_raw[i:i + sequence_length]
    counter = Counter(y_seq)
    most_common_class, count = counter.most_common(1)[0]
    if count >= 4:  # au moins 4 sur 5 identiques
        X.append(x_seq)
        y.append(most_common_class)

X = np.array(X)
y = np.array(y)

# Vérification du nombre d'échantillons
print(f"Nombre de séquences valides : {len(X)}")
if len(X) < 2:
    raise ValueError("Pas assez de séquences valides pour entraîner le modèle. Vérifie ton fichier Excel.")

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

# === Model ===
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train ===
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=100, batch_size=32)

# === Prédictions ===
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# === Évaluation ===
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

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

# === Save outputs ===
save_folder = "C:/Users/21653/Desktop/python/these/final"
os.makedirs(save_folder, exist_ok=True)
conf_matrix_df.to_excel(os.path.join(save_folder, "classification_report.xlsx"))
class_report_df.to_excel(os.path.join(save_folder, "accuracy_metrics.xlsx"))

plt.figure(figsize=(10, 6))

train_accuracy = [0] + history.history['accuracy']
val_accuracy = [0] + history.history['val_accuracy']

test_accuracy = [0] + [accuracy] * len(train_accuracy)  
plt.plot(train_accuracy, label='Training Accuracy', linestyle='solid', color='blue')
plt.plot(val_accuracy, label='Validation Accuracy', linestyle='solid', color='red')
plt.plot(test_accuracy, label='Test Accuracy', linestyle='-', color='green')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training, Validation & Test Accuracy')

plt.legend()
plt.grid(True)

plt.savefig(os.path.join(save_folder, "accuracy_plot.png"))
plt.show()



plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "confusion_matrix.png"))
plt.show()

# === Vérifie la distribution des classes ===
print("Distribution des classes dans y_train :")
print(pd.Series(y_train).value_counts())
print("Distribution des classes dans y_test :")
print(pd.Series(y_test).value_counts())

# === Sauvegarde du modèle ===
model.save(os.path.join(save_folder, "lstm_model.h5"))
