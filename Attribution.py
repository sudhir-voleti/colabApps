import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, HTML

sns.set_theme(style="whitegrid")

def run_attribution_app():
    """Renders the Multi-Channel Attribution Engine for Google Colab."""
    
    out = widgets.Output()
    upload_w = widgets.FileUpload(accept='.csv', multiple=False)
    
    seq_col_w = widgets.Dropdown(description='Sequence Col:')
    conv_col_w = widgets.Dropdown(description='Converted Col:')
    
    # Financial Inputs for EduGrowth Caselet B
    cpc_paid_w = widgets.FloatText(value=12.0, description='Paid Ad CPC ($):')
    cost_blog_w = widgets.FloatText(value=2000.0, description='Blog Cost ($):')
    signups_blog_w = widgets.IntText(value=500, description='Blog Signups:')
    
    calc_btn = widgets.Button(description='Run Attribution Analysis', button_style='primary', icon='chart-line')
    
    state = {'df': None}

    def on_upload_change(change):
        if not upload_w.value:
            return
        
        uploaded_file = list(upload_w.value.values())[0] if isinstance(upload_w.value, dict) else upload_w.value[0]
        content = uploaded_file['content'] if isinstance(uploaded_file, dict) else uploaded_file.content
        
        state['df'] = pd.read_csv(io.BytesIO(content))
        df = state['df']
        
        cols = list(df.columns)
        seq_col_w.options = cols
        conv_col_w.options = cols
        
        # Smart column selection
        for col in cols:
            lcol = col.lower()
            if 'sequence' in lcol or 'touch' in lcol or 'path' in lcol:
                seq_col_w.value = col
            if 'conv' in lcol or 'target' in lcol:
                conv_col_w.value = col

    upload_w.observe(on_upload_change, names='value')

    def parse_attribution(df, seq_col, conv_col):
        """Calculates First-Click, Last-Click, Linear, and Position-Based (U-Shaped) credits."""
        converters = df[df[conv_col].astype(str).str.strip().str.lower().isin(['1', 'yes', 'true'])].copy()
        
        models = {'First_Click': {}, 'Last_Click': {}, 'Linear': {}, 'Position_Based': {}}
        
        for seq in converters[seq_col].dropna():
            # Clean touchpoints list
            touches = [t.strip() for t in str(seq).split(',')]
            n_touches = len(touches)
            if n_touches == 0:
                continue
            
            # 1. First Click
            first = touches[0]
            models['First_Click'][first] = models['First_Click'].get(first, 0) + 1.0
            
            # 2. Last Click
            last = touches[-1]
            models['Last_Click'][last] = models['Last_Click'].get(last, 0) + 1.0
            
            # 3. Linear
            linear_credit = 1.0 / n_touches
            for t in touches:
                models['Linear'][t] = models['Linear'].get(t, 0) + linear_credit
                
            # 4. Position-Based (40 / 20 / 40)
            if n_touches == 1:
                models['Position_Based'][touches[0]] = models['Position_Based'].get(touches[0], 0) + 1.0
            elif n_touches == 2:
                models['Position_Based'][touches[0]] = models['Position_Based'].get(touches[0], 0) + 0.5
                models['Position_Based'][touches[1]] = models['Position_Based'].get(touches[1], 0) + 0.5
            else:
                models['Position_Based'][touches[0]] = models['Position_Based'].get(touches[0], 0) + 0.4
                models['Position_Based'][touches[-1]] = models['Position_Based'].get(touches[-1], 0) + 0.4
                mid_credit = 0.2 / (n_touches - 2)
                for t in touches[1:-1]:
                    models['Position_Based'][t] = models['Position_Based'].get(t, 0) + mid_credit

        attr_df = pd.DataFrame(models).fillna(0).round(2)
        attr_df.index.name = 'Channel'
        return attr_df.reset_index()

    def on_calc_clicked(b):
        with out:
            out.clear_output()
            df = state['df']
            if df is None:
                print("Please upload edugrowth_attribution_data.csv first.")
                return
            
            s_col = seq_col_w.value
            c_col = conv_col_w.value
            
            attr_summary = parse_attribution(df, s_col, c_col)
            
            print("=== ATTRIBUTED CONVERSIONS BY MODEL ===")
            display(attr_summary)
            
            # Calculate Cost Efficiency for Blog vs LinkedIn/Paid Ad
            cpc_paid = cpc_paid_w.value
            cost_per_blog_signup = cost_blog_w.value / signups_blog_w.value  # $4 per signup
            
            print("\n=== COST EFFICIENCY ANALYSIS (CPAC) ===")
            print(f"Assumed Cost Per Signup/Click: Paid Ad = ${cpc_paid:.2f} | Blog = ${cost_per_blog_signup:.2f}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            melted = attr_summary.melt(id_vars='Channel', var_name='Model', value_name='Attributed_Conversions')
            
            sns.barplot(data=melted, x='Channel', y='Attributed_Conversions', hue='Model', palette='magma', ax=ax)
            ax.set_title("Attribution Model Comparison: Last-Click vs Position-Based", fontsize=14, fontweight='bold')
            ax.set_ylabel("Attributed Conversions", fontweight='bold')
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.show()

    calc_btn.on_click(on_calc_clicked)

    ui_layout = widgets.VBox([
        widgets.HBox([upload_w]),
        widgets.HBox([seq_col_w, conv_col_w]),
        widgets.HBox([cpc_paid_w, cost_blog_w, signups_blog_w]),
        widgets.HBox([calc_btn]),
        out
    ])
    
    display(HTML("<h3>Multi-Channel Attribution Engine (EduGrowth Caselet B)</h3>"))
    display(ui_layout)

run_attribution_app()
