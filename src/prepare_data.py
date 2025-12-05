import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Helper functions
def load_excel(path, original_col, preferred_col):
    df = pd.read_excel(path)

    if original_col not in df.columns or preferred_col not in df.columns:
        raise ValueError(
            f"Columns not found. Available: {list(df.columns)} | "
            f"Expected: '{original_col}' and '{preferred_col}'"
        )

    df = df[[original_col, preferred_col]].dropna()
    df[original_col] = df[original_col].astype(str).str.strip()
    df[preferred_col] = df[preferred_col].astype(str).str.strip()

    df = df.rename(columns={original_col: "original", preferred_col: "preferred"})
    return df

def export_openai(df, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "messages": [
                    {"role": "user",
                     "content": f"Rewrite this clearly:\n\n{row['original']}"},
                    {"role": "assistant", "content": row["preferred"]},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def export_qlora(df, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "instruction": "Rewrite this clearly.",
                "input": row["original"],
                "output": row["preferred"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Main functions
def split_data(
    input_path, original_col = "Original", preferred_col = "Preferred", 
    test_size=0.1, save = False, train_path = None, test_path = None
):
    # Split data into train and test sets
    df = load_excel(input_path, original_col, preferred_col)
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        shuffle=True, random_state=42
    )

    if save:
        train_df.to_excel(train_path, index=False)
        test_df.to_excel(test_path, index=False)
        print("Saved split data")
    return train_df, test_df

def export_jsonl(input_data, original_col, preferred_col, type, output_path):
    # Export data to jsonl format for OpenAI or QLoRA fine-tuning
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        df = load_excel(input_data, original_col, preferred_col)

    if type == "openai":
        export_openai(df, output_path)
    if type == "qlora":
        export_qlora(df, output_path)
    print("Saved jsonl data")

if __name__ == "__main__":
    train, test = split_data(
        input_path="data/raw/obligations.xlsx",
        original_col="Original", preferred_col="Preferred",
        test_size=0.1
    )
    export_jsonl(
        input_data=train,
        output_path="data/processed/qlora_train.jsonl",
        original_col="Original", preferred_col="Preferred",
        type="qlora"
    )