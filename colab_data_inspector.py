import io
import re
import pandas as pd
from google.colab import files

def inspect_and_audit():
    """
    Interactively uploads CSV files in Google Colab, performs automated 
    data quality audits, detects primary/foreign keys, and outputs a clean, 
    markdown-formatted summary ready for AI Copilot prompting.
    """
    print("=================================================================")
    print("   SESSION 05: ENTERPRISE DATA QUALITY & SCHEMA INSPECTOR TOOL   ")
    print("=================================================================")
    print("Please click 'Choose Files' below and select all exported CSV files.\n")
    
    # 1. Trigger Colab File Upload Widget
    uploaded = files.upload()
    
    if not uploaded:
        print("\n[!] No files uploaded. Please re-run inspect_and_audit() to try again.")
        return

    dfs = {}
    
    # 2. Parse and Load CSVs into DataFrames
    for filename, content in uploaded.items():
        try:
            dfs[filename] = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            print(f"[!] Error reading {filename}: {e}")

    print("\n" + "=" * 65)
    print("           AUTOMATED DATA QUALITY & AUDIT REPORT                 ")
    print("=================================================================\n")

    markdown_prompt_block = []
    markdown_prompt_block.append("### ENTERPRISE DATASET SCHEMA & AUDIT SUMMARY\n")

    # 3. Perform Deep Data Quality Audit & Schema Analysis per File
    for filename, df in dfs.items():
        n_rows, n_cols = df.shape
        cols = list(df.columns)
        
        # A. Key Candidate Detection
        primary_keys = []
        foreign_keys = []
        
        for col in cols:
            # Check for candidate keys (strings/ints with '_id' or 'id')
            if 'id' in col.lower() or col.lower().endswith('_id') or col.lower().startswith('id_'):
                unique_count = df[col].nunique(dropna=True)
                if unique_count == n_rows:
                    primary_keys.append(col)
                else:
                    foreign_keys.append(f"{col} ({unique_count} unique values across {n_rows} rows -> 1:N Log)")

        # B. Missing Value & Null Diagnostics
        missing_dict = df.isnull().sum().to_dict()
        missing_cols = {col: count for col, count in missing_dict.items() if count > 0}
        
        # C. Data Quality Anomaly & Type Diagnostics
        anomalies = []
        
        for col in cols:
            # Detect Numeric Strings with Currency Symbols or Commas
            if df[col].dtype == 'object':
                sample_vals = df[col].dropna().astype(str).head(10).tolist()
                if any(re.search(r'[₹$%,]', val) for val in sample_vals):
                    anomalies.append(f"Column '{col}' contains formatted numeric strings (currency symbols or commas). Clean before modeling.")
                
                # Detect Trailing/Leading Whitespace
                if any(val != val.strip() for val in sample_vals):
                    anomalies.append(f"Column '{col}' has whitespace padding in string values.")

            # Detect Low-Variance Constant Columns
            if df[col].nunique(dropna=False) == 1:
                anomalies.append(f"Column '{col}' is constant (only 1 unique value across all rows). Consider dropping.")

        # D. Build Markdown Block for AI Copilot Prompting
        file_summary = f"#### FILE: `{filename}` ({n_rows} rows x {n_cols} columns)\n"
        file_summary += f"- **Primary Key Candidate(s) (100% Unique):** {', '.join([f'`{pk}`' for pk in primary_keys]) if primary_keys else 'None detected'}\n"
        file_summary += f"- **Foreign Key Candidate(s) (Repeated/1:N):** {', '.join([f'`{fk}`' for fk in foreign_keys]) if foreign_keys else 'None detected'}\n"
        
        # List Schema Variables with Types
        schema_list = [f"`{col}` ({df[col].dtype})" for col in cols]
        file_summary += f"- **Schema Columns ({n_cols}):** {', '.join(schema_list)}\n"
        
        # Missing Values Report
        if missing_cols:
            missing_str = ", ".join([f"`{k}`: {v} missing ({round(v/n_rows*100, 1)}%)" for k, v in missing_cols.items()])
            file_summary += f"- **Missing Values Detected:** {missing_str} -> *(Requires fillna or k-NN Imputation)*\n"
        else:
            file_summary += "- **Missing Values:** None (0% missing)\n"

        # Quality Warnings
        if anomalies:
            file_summary += "- **Data Quality Warnings:**\n"
            for warning in anomalies:
                file_summary += f"  * [!] {warning}\n"
        else:
            file_summary += "- **Data Quality Check:** Clean (No formatting or constant-column anomalies detected)\n"

        file_summary += "\n" + "-" * 50 + "\n"
        
        # Print directly to Colab Output
        print(file_summary)
        markdown_prompt_block.append(file_summary)

    # 4. Print Copy-Paste Container for AI Prompting
    full_prompt_str = "\n".join(markdown_prompt_block)
    
    print("\n=================================================================")
    print("   AI COPILOT PROMPT BLOCK (COPY & PASTE EVERYTHING BELOW)       ")
    print("=================================================================\n")
    print("```markdown")
    print(full_prompt_str)
    print("```")
    print("\n=================================================================")
    print("Copy the block above and paste it directly into your AI Copilot!")
    print("=================================================================")

# Auto-run entry point if executed directly
if __name__ == "__main__":
    inspect_and_audit()
