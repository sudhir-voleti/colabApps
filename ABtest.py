import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import ipywidgets as widgets
from IPython.display import display, HTML

# Clean Seaborn styling
sns.set_theme(style="whitegrid")

def run_ab_app():
    """Renders the main generic A/B testing ipywidgets interface inside Google Colab."""
    
    # ---------------------------------------------------------
    # TAB 1: SUMMARY CALCULATOR (N & Conversions)
    # ---------------------------------------------------------
    out_summary = widgets.Output()
    
    nA_w = widgets.IntText(value=1000, description='Control N:')
    xA_w = widgets.IntText(value=150, description='Control Conv:')
    nB_w = widgets.IntText(value=1000, description='Variant N:')
    xB_w = widgets.IntText(value=190, description='Variant Conv:')
    alpha_w = widgets.FloatSlider(value=0.05, min=0.01, max=0.10, step=0.01, description='Alpha (α):')
    calc_btn = widgets.Button(description='Calculate Z-Test', button_style='primary', icon='calculator')

    def on_calc_clicked(b):
        with out_summary:
            out_summary.clear_output()
            nA, xA = nA_w.value, xA_w.value
            nB, xB = nB_w.value, xB_w.value
            alpha = alpha_w.value
            
            if nA <= 0 or nB <= 0 or xA > nA or xB > nB:
                print("Error: Invalid sample sizes or conversion counts.")
                return
            
            crA = xA / nA
            crB = xB / nB
            rel_lift = ((crB - crA) / crA) * 100
            abs_lift = (crB - crA) * 100
            
            z_stat, p_val = proportions_ztest([xB, xA], [nB, nA], alternative='larger')
            
            is_sig = p_val < alpha
            color = "#28a745" if is_sig else "#dc3545"
            verdict = "Statistically Significant Uplift!" if is_sig else "No Statistically Significant Difference."
            
            html_res = f"""
            <div style="padding:15px; border-radius:8px; background-color:#f8f9fa; border-left: 5px solid {color}; margin-top:10px;">
                <h4 style="margin-0; color:#333;">Summary Hypothesis Test Results</h4>
                <ul>
                    <li><b>Control CR:</b> {crA:.2%} ({xA}/{nA})</li>
                    <li><b>Variant CR:</b> {crB:.2%} ({xB}/{nB})</li>
                    <li><b>Absolute Lift:</b> {abs_lift:+.2f} percentage points</li>
                    <li><b>Relative Lift:</b> {rel_lift:+.2f}%</li>
                    <li><b>Z-Statistic:</b> {z_stat:.4f}</li>
                    <li><b>P-Value:</b> {p_val:.4f}</li>
                </ul>
                <p style="font-weight:bold; color:{color}; font-size:15px;">Verdict: {verdict} (at α = {alpha})</p>
            </div>
            """
            display(HTML(html_res))

    calc_btn.on_click(on_calc_clicked)
    
    tab1_inputs = widgets.VBox([
        widgets.HBox([nA_w, xA_w]),
        widgets.HBox([nB_w, xB_w]),
        widgets.HBox([alpha_w, calc_btn])
    ])
    tab1_layout = widgets.VBox([tab1_inputs, out_summary])

    # ---------------------------------------------------------
    # TAB 2: DATASET MAPPER & SEGMENTED ANALYZER
    # ---------------------------------------------------------
    out_df = widgets.Output()
    upload_w = widgets.FileUpload(accept='.csv', multiple=False)
    
    seg_col_w = widgets.Dropdown(description='Segment Col:')
    var_col_w = widgets.Dropdown(description='Variant Col:')
    conv_col_w = widgets.Dropdown(description='Outcome Col:')
    
    ctrl_val_w = widgets.Dropdown(description='Control Val:')
    var_val_w = widgets.Dropdown(description='Target Var:')
    
    analyze_btn = widgets.Button(description='Run Full A/B Analysis', button_style='success', icon='chart-bar')
    
    state = {'df': None}

    def on_upload_change(change):
        if not upload_w.value:
            return
        
        # Parse uploaded file metadata across ipywidgets versions
        uploaded_file = list(upload_w.value.values())[0] if isinstance(upload_w.value, dict) else upload_w.value[0]
        content = uploaded_file['content'] if isinstance(uploaded_file, dict) else uploaded_file.content
        
        state['df'] = pd.read_csv(io.BytesIO(content))
        df = state['df']
        
        cols = list(df.columns)
        seg_col_w.options = ['(None / Aggregate)'] + cols
        var_col_w.options = cols
        conv_col_w.options = cols
        
        # Smart column guessing
        for col in cols:
            lcol = col.lower()
            if 'var' in lcol or 'group' in lcol:
                var_col_w.value = col
            if 'conv' in lcol or 'target' in lcol or 'outcome' in lcol:
                conv_col_w.value = col
            if 'seg' in lcol or 'channel' in lcol or 'source' in lcol:
                seg_col_w.value = col

    upload_w.observe(on_upload_change, names='value')

    def on_var_col_change(change):
        if state['df'] is not None and var_col_w.value in state['df'].columns:
            unique_vals = list(state['df'][var_col_w.value].dropna().unique())
            ctrl_val_w.options = unique_vals
            var_val_w.options = unique_vals
            if len(unique_vals) > 1:
                var_val_w.value = unique_vals[1]

    var_col_w.observe(on_var_col_change, names='value')

    def on_analyze_clicked(b):
        with out_df:
            out_df.clear_output()
            df = state['df']
            if df is None:
                print("Please upload a CSV file first.")
                return
            
            v_col = var_col_w.value
            c_col = conv_col_w.value
            s_col = seg_col_w.value
            ctrl_v = ctrl_val_w.value
            target_v = var_val_w.value
            
            # Ensure outcome is numeric (0 and 1)
            temp_df = df.copy()
            if temp_df[c_col].dtype == object:
                temp_df[c_col] = temp_df[c_col].astype(str).str.strip().str.lower().map({'1': 1, '0': 0, 'yes': 1, 'no': 0, 'true': 1, 'false': 0})
            
            # 1. Overall Comparison Table
            grouped = temp_df.groupby(v_col)[c_col].agg(['count', 'sum', 'mean']).reset_index()
            grouped.columns = [v_col, 'Total Users', 'Conversions', 'Conversion Rate']
            grouped['Conversion Rate (%)'] = (grouped['Conversion Rate'] * 100).round(2)
            
            print("=== OVERALL VARIANT PERFORMANCE ===")
            display(grouped)
            
            # 2. Chi-Squared Test of Independence
            ctable = pd.crosstab(temp_df[v_col], temp_df[c_col])
            chi2, p_chi, dof, _ = chi2_contingency(ctable)
            
            print(f"\n=== CHI-SQUARED TEST OF INDEPENDENCE ===")
            print(f"Chi2 Statistic: {chi2:.4f} | P-value: {p_chi:.4f} | DoF: {dof}")
            if p_chi < 0.05:
                print("Result: Statistically significant association between Variant and Conversion!")
            else:
                print("Result: No statistically significant association detected overall.")
                
            # 3. Segment Breakdown (if selected)
            if s_col != '(None / Aggregate)':
                print(f"\n=== SEGMENTED ANALYSIS BY '{s_col}' ===")
                seg_summary = temp_df.groupby([s_col, v_col])[c_col].agg(['count', 'mean']).reset_index()
                seg_summary['Conversion Rate (%)'] = (seg_summary['mean'] * 100).round(2)
                
                # Plotting
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    data=seg_summary,
                    x=s_col,
                    y='Conversion Rate (%)',
                    hue=v_col,
                    palette='viridis',
                    ax=ax
                )
                ax.set_title(f"A/B Test Conversion Rates Segmented by {s_col}", fontsize=14, fontweight='bold')
                ax.set_ylabel("Conversion Rate (%)", fontweight='bold')
                ax.set_xlabel(s_col, fontweight='bold')
                
                for p in ax.patches:
                    h = p.get_height()
                    if not np.isnan(h) and h > 0:
                        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                                    textcoords='offset points')
                
                plt.tight_layout()
                plt.show()

    analyze_btn.on_click(on_analyze_clicked)

    tab2_inputs = widgets.VBox([
        widgets.HBox([upload_w]),
        widgets.HBox([seg_col_w, var_col_w, conv_col_w]),
        widgets.HBox([ctrl_val_w, var_val_w, analyze_btn])
    ])
    tab2_layout = widgets.VBox([tab2_inputs, out_df])

    # Build Tabs UI
    tabs = widgets.Tab(children=[tab1_layout, tab2_layout])
    tabs.set_title(0, 'Summary Calculator')
    tabs.set_title(1, 'Dataset Analyzer')
    
    display(HTML("<h3>Generic A/B Testing & Hypothesis Testing Engine</h3>"))
    display(tabs)

# Trigger initialization when executed directly via exec()
run_ab_app()
