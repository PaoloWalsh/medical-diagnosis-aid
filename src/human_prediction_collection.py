import pandas as pd

# Load your dataset
input_file = '../data/heart_disease_clean.csv'
df = pd.read_csv(input_file)

# Collect human labels here
human_labels = []

print(f"\n--- Reviewing {len(df)} patient records ---\n")

# Loop through each row
for idx, row in df.iterrows():
    print(f"\n🩺 Patient {idx + 1} of {len(df)}")

    # Display as table using transpose
    for col in df.columns:
        if col in ['id', 'dataset', 'heart_disease_prediction', 'sick']:
            continue
        print(f"{col}: {row[col]}")

    # Ask for input
    while True:
        user_input = input("👉 Is this patient sick? (y/n) or type 'quit' to stop: ").strip().lower()
        if user_input in ['y', 'n', 'quit']:
            break
        else:
            print("❗ Invalid input. Type 'y', 'n', or 'quit'.")

    if user_input == 'quit':
        break

    human_labels.append(1 if user_input == 'y' else 0)

# Save output if any
if human_labels:
    labeled_df = df.iloc[:len(human_labels)].copy()
    labeled_df['human_prediction'] = human_labels

    output_file = '../data/heart_disease_human_prediction.csv'
    labeled_df.to_csv(output_file, index=False)

    print(f"\n✅ Done. {len(labeled_df)} rows saved to '{output_file}'.")
else:
    print("\n⚠️ No data labeled. Nothing saved.")
