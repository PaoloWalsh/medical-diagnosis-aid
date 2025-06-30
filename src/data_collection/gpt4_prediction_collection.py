import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your dataset
input_file = '../data/heart_disease_clean.csv'
df = pd.read_csv(input_file)

# Prepare label list
gpt_labels = []

print(f"\n--- Reviewing {len(df)} patient records ---\n")

# Loop through each row
for idx, row in df.iterrows():
    
    # Construct patient string from row
    patient_string = ' '.join([
        f"{col}: {row[col]}" for col in df.columns
        if col not in ['id', 'dataset', 'heart_disease_prediction', 'sick']
    ])

    prompt = f"Without searching on the internet answer with a yes or no (y/n), does this patient have heart disease? {patient_string}"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        message = response.choices[0].message.content.strip().lower()
        label = 1 if message.startswith('y') else 0
    except OpenAI.error.OpenAIError as e:
        print(f"⚠️ Error on row {idx}: {e}")
        label = None
    gpt_labels.append(label)

# Save results
if any(label is not None for label in gpt_labels):
    labeled_df = df.iloc[:len(gpt_labels)].copy()
    labeled_df['gpt_prediction'] = gpt_labels

    output_file = '../data/heart_disease_gpt_prediction.csv'
    labeled_df.to_csv(output_file, index=False)

    print(f"\n✅ Done. {len(labeled_df)} rows saved to '{output_file}'.")
else:
    print("\n⚠️ No data labeled. Nothing saved.")
