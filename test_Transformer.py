import pandas as pd

# Dictionnaire des noms de classes
class_names = {
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

def detect_class(group):
    thetas = {f'theta{i}': group[f'theta{i}'].values for i in range(1, 7)}

    def is_constant_from(values, start_idx):
        return all(v == values[start_idx] for v in values[start_idx:])

    for i in range(1, 7):
        theta_values = thetas[f'theta{i}']
        for idx, class_offset in zip([1, 2, 3], [0, 1, 2]):
            if is_constant_from(theta_values, idx):
                base_class = (i - 1) * 3 + 1
                class_num = base_class + class_offset
                return class_names[class_num]

    return class_names[19]

def process_dataset(file_path, output_path):
    df = pd.read_excel(file_path)
    groups = [df.iloc[i:i+5] for i in range(0, len(df), 5)]

    results = []
    for group in groups:
        class_label = detect_class(group)
        group = group.copy()
        group["Class"] = [class_label] * 5
        results.append(group)

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_excel(output_path, index=False)
    print(f"Fichier traité sauvegardé : {output_path}")

# Exemple d'utilisation
process_dataset("C:/Users/21653/Downloads/these/datafinalelstmdetest.xlsx", "C:/Users/21653/Downloads/these/dataset.xlsx")
