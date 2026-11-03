import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import ipywidgets as widgets
from IPython.display import display, HTML

# Set clean white background style
sns.set_theme(style="whitegrid")

def run_generic_ab_app():
    """Renders the redesigned, 3-tab generic A/B Testing & Hypothesis Engine for Google Colab."""
    
    # State dictionary to persist dataset and configuration across tabs
    state = {
        'df': None,
        'y_col': None,
        'x_col': None,
        'val_a': None,
        'val_b': None,
        'seg_col': None,
        'is_binary': False
    }

    # Helper for horizontal line dividers
    def make_hr():
        return widgets.HTML('<hr style="margin: 10px 0; border: 0; border-top: 1px solid #ccc;">')

    # =========================================================
    # TAB 1: DATA UPLOAD & PREVIEW
    # =========================================================
    upload_w = widgets.FileUpload(accept='.csv', multiple=False)
    out_tab1 = widgets.Output()

    def on_upload_change(change):
        if not upload_w.value:
            return
        
        with out_tab1:
            out_tab1.clear_output()
            # Handle ipywidgets file upload schema across versions
            uploaded_file = list(upload_w.value.values())[0] if isinstance(upload_w.value, dict) else upload_w.value[0]
            content = uploaded_file['content'] if isinstance(uploaded_file, dict) else uploaded_file.content
            
            try:
                state['df'] = pd.read_csv(io.BytesIO(content))
                df = state['df']
                
                # Update dropdown options in Tab 2
                cols = list(df.columns)
                y_dropdown.options = cols
                x_dropdown.options = cols
                seg_dropdown.options = ['(None)'] + cols
                
                print(f"Dataset successfully loaded: {df.shape[0]} rows × {df.shape[1]} columns.\n")
                print("=== DATASET PREVIEW (FIRST 10 ROWS) ===")
                display(df.head(10))
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")

    upload_w.observe(on_upload_change, names='value')
    tab1_layout = widgets.VBox([
        widgets.HTML("<h4>Step 1: Upload Dataset</h4>"),
        upload_w,
        make_hr(),
        out_tab1
    ])

    # =========================================================
    # TAB 2: VARIABLE SELECTION & DESCRIPTIVE STATS
    # =========================================================
    y_dropdown = widgets.Dropdown(description='Outcome (Y):')
    x_dropdown = widgets.Dropdown(description='Grouping (X):')
    val_a_dropdown = widgets.Dropdown(description='Group A (Ref):')
    val_b_dropdown = widgets.Dropdown(description='Group B (Comp):')
    seg_dropdown = widgets.Dropdown(description='Segment (Opt):')
    
    lock_btn = widgets.Button(description='Lock Selection & Compute Summary', button_style='primary', icon='lock')
    out_tab2 = widgets.Output()

    def on_x_change(change):
        """Update Group A and Group B value choices when Grouping Variable X changes."""
        if state['df'] is not None and x_dropdown.value in state['df'].columns:
            unique_vals = list(state['df'][x_dropdown.value].dropna().unique())
            val_a_dropdown.options = unique_vals
            val_b_dropdown.options = unique_vals
            if len(unique_vals) > 1:
                val_a_dropdown.value = unique_vals[0]
                val_b_dropdown.value = unique_vals[1]

    x_dropdown.observe(on_x_change, names='value')

    def on_lock_clicked(b):
        with out_tab2:
            out_tab2.clear_output()
            df = state['df']
            if df is None:
                print("Please upload a dataset in Tab 1 first.")
                return
            
            y_col = y_dropdown.value
            x_col = x_dropdown.value
            val_a = val_a_dropdown.value
            val_b = val_b_dropdown.value
            seg_col = seg_dropdown.value
            
            if val_a == val_b:
                print("Error: Group A and Group B must be different unique values.")
                return
            
            # Save selections to state
            state['y_col'] = y_col
            state['x_col'] = x_col
            state['val_a'] = val_a
            state['val_b'] = val_b
            state['seg_col'] = seg_col if seg_col != '(None)' else None
            
            # Detect if outcome Y is binary numeric/string
            y_series = df[y_col].dropna()
            unique_y = y_series.unique()
            
            # Auto-clean binary strings to 0 and 1 if applicable
            if len(unique_y) == 2 and set(unique_y).issubset({0, 1, '0', '1', 'Yes', 'No', 'yes', 'no', 'TRUE', 'FALSE', 'true', 'false'}):
                state['is_binary'] = True
                mapping = {1:1, 0:0, '1':1, '0':0, 'Yes':1, 'No':0, 'yes':1, 'no':0, 'TRUE':1, 'FALSE':0, 'true':1, 'false':0}
                state['df'][y_col] = state['df'][y_col].map(mapping)
            else:
                state['is_binary'] = pd.api.types.is_numeric_dtype(y_series) and len(unique_y) == 2
            
            # Filter dataset for Group A and Group B only
            subset = df[df[x_col].isin([val_a, val_b])].copy()
            
            print("Selection Locked Successfully!\n")
            print("=== DESCRIPTIVE STATISTICS BY GROUP ===")
            
            desc = subset.groupby(x_col)[y_col].agg(
                N='count',
                Mean='mean',
                Std_Dev='std',
                Min='min',
                Max='max'
            ).reset_index()
            desc['Mean'] = desc['Mean'].round(4)
            desc['Std_Dev'] = desc['Std_Dev'].round(4)
            display(desc)
            
            # Segment descriptive stats if selected
            if state['seg_col']:
                print(f"\n=== DESCRIPTIVE STATISTICS STRATIFIED BY SEGMENT ({state['seg_col']}) ===")
                seg_desc = subset.groupby([state['seg_col'], x_col])[y_col].agg(
                    N='count',
                    Mean='mean',
                    Std_Dev='std'
                ).reset_index()
                seg_desc['Mean'] = seg_desc['Mean'].round(4)
                seg_desc['Std_Dev'] = seg_desc['Std_Dev'].round(4)
                display(seg_desc)

    lock_btn.on_click(on_lock_clicked)
    
    tab2_layout = widgets.VBox([
        widgets.HTML("<h4>Step 2: Map Variables & Confirm Selection</h4>"),
        widgets.HBox([y_dropdown, x_dropdown]),
        widgets.HBox([val_a_dropdown, val_b_dropdown]),
        widgets.HBox([seg_dropdown, lock_btn]),
        make_hr(),
        out_tab2
    ])

    # =========================================================
    # TAB 3: MODEL RESULTS & VISUALIZATION
    # =========================================================
    run_test_btn = widgets.Button(description='Run Model Analysis', button_style='success', icon='chart-bar')
    out_tab3 = widgets.Output()

    def on_run_test_clicked(b):
        with out_tab3:
            out_tab3.clear_output()
            df = state['df']
            y_col = state['y_col']
            x_col = state['x_col']
            val_a = state['val_a']
            val_b = state['val_b']
            seg_col = state['seg_col']
            
            if df is None or y_col is None:
                print("Please complete and lock variable selections in Tab 2 first.")
                return
            
            # Filter data
            data_a = df[df[x_col] == val_a][y_col].dropna()
            data_b = df[df[x_col] == val_b][y_col].dropna()
            
            nA, meanA = len(data_a), data_a.mean()
            nB, meanB = len(data_b), data_b.mean()
            
            diff = meanB - meanA
            pct_lift = (diff / meanA * 100) if meanA != 0 else np.nan
            
            # Perform appropriate test (Z-test for Proportions vs t-test for Means)
            if state['is_binary']:
                test_type = "Two-Sample Proportion Z-Test"
                count = [data_b.sum(), data_a.sum()]
                nobs = [nB, nA]
                stat, p_val = proportions_ztest(count, nobs)
                stat_label = "Z-Statistic"
            else:
                test_type = "Two-Sample Independent t-Test"
                stat, p_val = ttest_ind(data_b, data_a, equal_var=False)
                stat_label = "t-Statistic"
                
            is_sig = p_val < 0.05
            color = "#28a745" if is_sig else "#dc3545"
            verdict = "Statistically Significant Difference (p < 0.05)" if is_sig else "No Statistically Significant Difference (p ≥ 0.05)"
            
            print(f"=== STATISTICAL ANALYSIS RESULTS ({test_type}) ===")
            
            res_df = pd.DataFrame({
                'Group': [f"Group A ({val_a})", f"Group B ({val_b})", "Difference (B - A)"],
                'N': [nA, nB, nA + nB],
                'Metric Value (Mean/Rate)': [round(meanA, 4), round(meanB, 4), round(diff, 4)],
                'Relative Lift (%)': ['-', '-', f"{pct_lift:+.2f}%" if not np.isnan(pct_lift) else '-']
            })
            display(res_df)
            
            html_verdict = f"""
            <div style="padding:12px; border-radius:6px; background-color:#f8f9fa; border-left: 5px solid {color}; margin:15px 0;">
                <b>Test Executed:</b> {test_type}<br>
                <b>{stat_label}:</b> {stat:.4f} | <b>P-Value:</b> {p_val:.4f}<br>
                <span style="color:{color}; font-weight:bold;">Verdict: {verdict}</span>
            </div>
            """
            display(HTML(html_verdict))
            
            # Segment interaction analysis if segment variable was specified
            if seg_col:
                print(f"=== SEGMENTED BREAKDOWN ACROSS '{seg_col}' ===")
                seg_results = []
                subset = df[df[x_col].isin([val_a, val_b])].copy()
                
                for seg_val, group in subset.groupby(seg_col):
                    gA = group[group[x_col] == val_a][y_col].dropna()
                    gB = group[group[x_col] == val_b][y_col].dropna()
                    
                    if len(gA) > 0 and len(gB) > 0:
                        mA, mB = gA.mean(), gB.mean()
                        d = mB - mA
                        if state['is_binary']:
                            s, p = proportions_ztest([gB.sum(), gA.sum()], [len(gB), len(gA)])
                        else:
                            s, p = ttest_ind(gB, gA, equal_var=False)
                        
                        seg_results.append({
                            seg_col: seg_val,
                            f'N ({val_a})': len(gA),
                            f'Mean ({val_a})': round(mA, 4),
                            f'N ({val_b})': len(gB),
                            f'Mean ({val_b})': round(mB, 4),
                            'Diff (B - A)': round(d, 4),
                            'P-Value': round(p, 4),
                            'Significant?': 'Yes' if p < 0.05 else 'No'
                        })
                
                seg_res_df = pd.DataFrame(seg_results)
                display(seg_res_df)

            # Chart Visualization
            print("\n=== METRIC COMPARISON CHART ===")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            subset_plot = df[df[x_col].isin([val_a, val_b])].copy()
            
            if seg_col:
                plot_data = subset_plot.groupby([seg_col, x_col])[y_col].mean().reset_index()
                sns.barplot(data=plot_data, x=seg_col, y=y_col, hue=x_col, palette='viridis', ax=ax)
                ax.set_title(f"Metric ({y_col}) by {x_col} Across Segments ({seg_col})", fontweight='bold')
            else:
                plot_data = subset_plot.groupby(x_col)[y_col].mean().reset_index()
                sns.barplot(data=plot_data, x=x_col, y=y_col, palette='viridis', ax=ax)
                ax.set_title(f"Metric ({y_col}) Comparison by {x_col}", fontweight='bold')
                
            ax.set_ylabel(f"Mean {y_col}", fontweight='bold')
            ax.set_xlabel(x_col, fontweight='bold')
            
            for p in ax.patches:
                h = p.get_height()
                if not np.isnan(h) and h > 0:
                    ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                                textcoords='offset points')
                    
            plt.tight_layout()
            plt.show()

    run_test_btn.on_click(on_run_test_clicked)
    
    tab3_layout = widgets.VBox([
        widgets.HTML("<h4>Step 3: Execute Statistical Model & Inspect Visuals</h4>"),
        run_test_btn,
        make_hr(),
        out_tab3
    ])

    # Build Master Tab UI
    tabs = widgets.Tab(children=[tab1_layout, tab2_layout, tab3_layout])
    tabs.set_title(0, '1. Data & Preview')
    tabs.set_title(1, '2. Variables & Descriptives')
    tabs.set_title(2, '3. Model Results')
    
    display(HTML("<h3>Generic Hypothesis & A/B Testing Engine</h3>"))
    display(tabs)

# Trigger app when executed directly via exec()
run_generic_ab_app()
