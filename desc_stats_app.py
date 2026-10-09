# ============================================================================
# Session 01 - Descriptive Statistics App (Generic CSV Upload Engine)
# Repo : github.com/sudhir-voleti/colabApps
# Launch code in Colab:
#   import requests
#   exec(requests.get("https://raw.githubusercontent.com/sudhir-voleti/colabApps/main/desc_stats_app.py").text)
#   launch_app()
# ============================================================================

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

def launch_app():
    # 1. UI Components Setup
    upload_widget = widgets.FileUpload(
        accept='.csv',
        multiple=False,
        description='Upload CSV',
        button_style='primary',
        icon='upload'
    )
    
    out_main = widgets.Output()
    
    print("=" * 64)
    print("  Session 01: Interactive Descriptive Statistics Engine")
    print("=" * 64)
    print("Please upload your CSV file to begin analysis:\n")
    display(upload_widget)
    display(out_main)
    
    # 2. Main Callback on File Upload
    def on_file_upload(change):
        with out_main:
            clear_output()
            if not upload_widget.value:
                return
            
            # Extract raw uploaded file bytes
            uploaded_file = list(upload_widget.value.values())[0] if isinstance(upload_widget.value, dict) else upload_widget.value[0]
            content = uploaded_file['content'] if isinstance(uploaded_file, dict) else uploaded_file.content
            
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception as e:
                print(f"❌ Error reading CSV file: {e}")
                return
            
            print(f"✅ File successfully loaded! Data Shape: {df.shape[0]} rows × {df.shape[1]} columns.\n")
            
            # --- Auto-Detection of Variable Types ---
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            # Heuristic: Move low-cardinality numerics (<=10 unique values, e.g. Month, Tier ID) to categoricals if helpful
            for col in num_cols[:]:
                if df[col].nunique() <= 5 and col not in cat_cols:
                    cat_cols.append(col)

            if not num_cols and not cat_cols:
                print("❌ No readable numeric or categorical columns found.")
                return

            # --- TAB 1: NUMERIC ANALYTICS ---
            out_tab1 = widgets.Output()
            num_dropdown = widgets.Dropdown(options=num_cols, description='Metric:') if num_cols else None
            
            def render_numeric_tab(change=None):
                with out_tab1:
                    clear_output()
                    if not num_cols:
                        print("No numeric columns detected in this dataset.")
                        return
                    
                    col = num_dropdown.value
                    s = df[col].dropna()
                    
                    mean_val = s.mean()
                    med_val = s.median()
                    std_val = s.std()
                    cv_val = std_val / mean_val if mean_val != 0 else np.nan
                    mode_val = s.mode()[0] if not s.mode().empty else np.nan
                    
                    summary_df = pd.DataFrame({
                        "Metric": [col],
                        "Mean": [round(mean_val, 2)],
                        "Median": [round(med_val, 2)],
                        "Mode": [round(mode_val, 2)],
                        "Std Dev (Wobble)": [round(std_val, 2)],
                        "CV (Wobble/Rupee)": [round(cv_val, 3)],
                        "Min": [round(s.min(), 2)],
                        "Max": [round(s.max(), 2)],
                        "Count": [int(s.count())]
                    })
                    
                    print(f"--- Executive Numeric Summary: {col} ---")
                    display(summary_df)
                    
                    # Distribution Plot
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    s.plot(kind='hist', bins=20, alpha=0.5, color='#003366', density=True, ax=ax)
                    s.plot(kind='kde', color='#D97706', linewidth=2, ax=ax)
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean ({mean_val:.2f})')
                    ax.axvline(med_val, color='green', linestyle=':', linewidth=2, label=f'Median ({med_val:.2f})')
                    ax.set_title(f"Distribution & Skewness Check: {col}", fontsize=11, fontweight='bold')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

            if num_dropdown:
                num_dropdown.observe(render_numeric_tab, names='value')

            # --- TAB 2: CATEGORICAL ANALYTICS ---
            out_tab2 = widgets.Output()
            cat_dropdown = widgets.Dropdown(options=cat_cols, description='Category:') if cat_cols else None
            
            def render_categorical_tab(change=None):
                with out_tab2:
                    clear_output()
                    if not cat_cols:
                        print("No categorical columns detected in this dataset.")
                        return
                    
                    col = cat_dropdown.value
                    s = df[col].astype(str)
                    
                    counts = s.value_counts()
                    props = (s.value_counts(normalize=True) * 100).round(1)
                    
                    cat_summary = pd.DataFrame({
                        "Count": counts,
                        "Proportion (%)": props
                    })
                    
                    print(f"--- Categorical Breakdown: {col} ---")
                    display(cat_summary)
                    
                    # Category Bar Chart
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    props.plot(kind='bar', color='#003366', ax=ax)
                    ax.set_title(f"Proportions (%): {col}", fontsize=11, fontweight='bold')
                    ax.set_ylabel("Percentage (%)")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()

            if cat_dropdown:
                cat_dropdown.observe(render_categorical_tab, names='value')

            # --- TAB 3: GROUPED BREAKDOWN & CROSSTABS ---
            out_tab3 = widgets.Output()
            group_cat = widgets.Dropdown(options=cat_cols, description='Group (Cat):') if cat_cols else None
            group_target = widgets.Dropdown(options=num_cols + cat_cols, description='Target Var:')
            
            def render_grouped_tab(change=None):
                with out_tab3:
                    clear_output()
                    if not cat_cols:
                        print("Grouping requires at least one categorical column.")
                        return
                    
                    cat_var = group_cat.value
                    target_var = group_target.value
                    
                    if target_var in num_cols:
                        print(f"--- Grouped Metric Summary: {target_var} by {cat_var} ---")
                        res = df.groupby(cat_var)[target_var].agg(
                            Count='count', Mean='mean', Median='median', Std_Dev='std'
                        ).round(2)
                        display(res)
                    else:
                        print(f"--- Row % Crosstab: {cat_var} vs {target_var} ---")
                        ct = pd.crosstab(df[cat_var], df[target_var], normalize='index') * 100
                        display(ct.round(1))

            if group_cat and group_target:
                group_cat.observe(render_grouped_tab, names='value')
                group_target.observe(render_grouped_tab, names='value')

            # Build App Layout inside Tabs
            tab_ui = widgets.Tab()
            
            # Assemble Tab 1 UI
            ui_tab1 = widgets.VBox([num_dropdown, out_tab1]) if num_dropdown else widgets.VBox([out_tab1])
            # Assemble Tab 2 UI
            ui_tab2 = widgets.VBox([cat_dropdown, out_tab2]) if cat_dropdown else widgets.VBox([out_tab2])
            # Assemble Tab 3 UI
            ui_tab3 = widgets.VBox([widgets.HBox([group_cat, group_target]), out_tab3]) if group_cat else widgets.VBox([out_tab3])
            
            tab_ui.children = [ui_tab1, ui_tab2, ui_tab3]
            tab_ui.set_title(0, 'Numeric Analytics')
            tab_ui.set_title(1, 'Categorical Analytics')
            tab_ui.set_title(2, 'Grouped & Crosstabs')
            
            display(tab_ui)
            
            # Initial renders
            render_numeric_tab()
            render_categorical_tab()
            render_grouped_tab()

    upload_widget.observe(on_file_upload, names='value')
