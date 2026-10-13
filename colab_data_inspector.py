import io
import re
import glob
import os
import pandas as pd
from google.colab import files

def inspect_and_audit():
    """
    Uploads CSVs, purges old files, and outputs a Markdown audit block 
    including 2-row preview samples for AI visual type checking.
    """
    for old_file in glob.glob("*.csv"):
        try:
            os.remove(old_file)
        except Exception:
            pass

    print("Select all CSV files to upload...")
    uploaded = files.upload()
    
    if not uploaded:
        print("[!] No files uploaded.")
        return

    dfs = {}
    for filename, content in uploaded.items():
        try:
            dfs[filename] = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            print(f"[!] Error reading {filename}: {e}")

    markdown_output = []
    markdown_output.append("### ENTERPRISE DATASET SCHEMA & AUDIT REPORT\n")

    for filename, df in dfs.items():
        n_rows, n_cols = df.shape
        cols = list(df.columns)
        
        primary_keys = []
        foreign_keys = []
        
        for col in cols:
            if 'id' in col.lower() or col.lower().endswith('_id') or col.lower().startswith('id_'):
                unique_count = df[col].nunique(dropna=True)
                if unique_count == n_rows:
                    primary_keys.append(col)
                else:
                    foreign_keys.append(f"{col} ({unique_count} unique / {n_rows} rows)")

        missing_dict = df.isnull().sum().to_dict()
        missing_cols = {col: count for col, count in missing_dict.items() if count > 0}
        
        anomalies = []
        for col in cols:
            if df[col].dtype == 'object':
                sample_vals = df[col].dropna().astype(str).head(10).tolist()
                if any(re.search(r'[₹$%,]', val) for val in sample_vals):
                    anomalies.append(f"Column '{col}' contains formatted numeric strings (currency/commas).")
                if any(val != val.strip() for val in sample_vals):
                    anomalies.append(f"Column '{col}' has whitespace padding.")
            if df[col].nunique(dropna=False) == 1:
                anomalies.append(f"Column '{col}' is constant (only 1 unique value).")

        file_block = f"#### FILE: `{filename}` ({n_rows} rows x {n_cols} cols)\n"
        file_block += f"- Primary Keys: {', '.join([f'`{pk}`' for pk in primary_keys]) if primary_keys else 'None'}\n"
        file_block += f"- Foreign Keys: {', '.join([f'`{fk}`' for fk in foreign_keys]) if foreign_keys else 'None'}\n"
        
        schema_list = [f"`{col}` ({df[col].dtype})" for col in cols]
        file_block += f"- Columns: {', '.join(schema_list)}\n"
        
        if missing_cols:
            missing_str = ", ".join([f"`{k}`: {v} missing ({round(v/n_rows*100, 1)}%)" for k, v in missing_cols.items()])
            file_block += f"- Missing Values: {missing_str}\n"
        else:
            file_block += "- Missing Values: None\n"

        if anomalies:
            file_block += "- Automated Quality Warnings:\n"
            for warning in anomalies:
                file_block += f"  * {warning}\n"
        else:
            file_block += "- Automated Quality Warnings: None\n"

        # Add 2-row sample preview for visual AI inspection
        file_block += "- **First 2 Rows Sample:**\n"
        file_block += "```text\n"
        file_block += df.head(2).to_string(index=False) + "\n"
        file_block += "```\n"

        file_block += "\n" + "-" * 40 + "\n"
        markdown_output.append(file_block)

    print("\n" * 2)
    print("=" * 10 + " COPY ONLY THE TEXT BELOW THIS LINE " + "=" * 10 + "\n")
    print("\n".join(markdown_output))

if __name__ == "__main__":
    inspect_and_audit()
