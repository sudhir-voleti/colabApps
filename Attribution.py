import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, HTML

sns.set_theme(style="whitegrid")

def run_generic_attribution_app():
    """Renders the redesigned 3-Tab Generic Multi-Channel Attribution Engine for Google Colab."""
    
    state = {'df': None, 'channels': [], 'cost_widgets': {}}
    
    style_opts = {'description_width': 'initial'}
    widget_layout = widgets.Layout(width='340px')

    def make_hr():
        return widgets.HTML('<hr style="margin: 10px 0; border: 0; border-top: 1px solid #ccc;">')

    # =========================================================
    # TAB 1: DATA & PREVIEW
    # =========================================================
    upload_w = widgets.FileUpload(accept='.csv', multiple=False)
    out_tab1 = widgets.Output()

    def on_upload_change(change):
        if not upload_w.value:
            return
        
        with out_tab1:
            out_tab1.clear_output()
            uploaded_file = list(upload_w.value.values())[0] if isinstance(upload_w.value, dict) else upload_w.value[0]
            content = uploaded_file['content'] if isinstance(uploaded_file, dict) else uploaded_file.content
            
            try:
                state['df'] = pd.read_csv(io.BytesIO(content))
                df = state['df']
                
                cols = list(df.columns)
                seq_col_w.options = cols
                conv_col_w.options = cols
                
                for col in cols:
                    lcol = col.lower()
                    if 'sequence' in lcol or 'touch' in lcol or 'path' in lcol:
                        seq_col_w.value = col
                    if 'conv' in lcol or 'target' in lcol:
                        conv_col_w.value = col
                
                update_channel_cost_inputs()
                
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
    # TAB 2: CONFIGURATION & COST INPUTS
    # =========================================================
    seq_col_w = widgets.Dropdown(description='Touchpoint Sequence Column:', style=style_opts, layout=widget_layout)
    conv_col_w = widgets.Dropdown(description='Conversion Indicator Column:', style=style_opts, layout=widget_layout)
    delim_w = widgets.Text(value=',', description='Sequence Delimiter:', style=style_opts, layout=widgets.Layout(width='200px'))
    
    cost_box = widgets.VBox([])
    lock_btn = widgets.Button(
        description='Confirm Mapping & Build Cost Matrix', 
        button_style='primary', 
        icon='lock',
        layout=widgets.Layout(width='auto', min_width='280px')
    )
    out_tab2 = widgets.Output()

    def update_channel_cost_inputs(*args):
        if state['df'] is None or not seq_col_w.value:
            return
        
        df = state['df']
        s_col = seq_col_w.value
        delim = delim_w.value or ','
        
        raw_sequences = df[s_col].dropna().astype(str)
        unique_channels = sorted(list(set(
            ch.strip() for seq in raw_sequences for ch in seq.split(delim) if ch.strip()
        )))
        
        state['channels'] = unique_channels
        state['cost_widgets'] = {}
        
        widget_list = [HTML("<b>Optional: Enter Unit Cost ($ per Acquisition / Touch) for Discovered Channels:</b>")]
        
        rows = []
        row_items = []
        for ch in unique_channels:
            w = widgets.FloatText(value=0.0, description=f"{ch}:", style=style_opts, layout=widgets.Layout(width='220px'))
            state['cost_widgets'][ch] = w
            row_items.append(w)
            if len(row_items) == 3:
                rows.append(widgets.HBox(row_items))
                row_items = []
        if row_items:
            rows.append(widgets.HBox(row_items))
            
        cost_box.children = tuple(widget_list + rows)

    seq_col_w.observe(update_channel_cost_inputs, names='value')
    delim_w.observe(update_channel_cost_inputs, names='value')

    def on_lock_clicked(b):
        with out_tab2:
            out_tab2.clear_output()
            if state['df'] is None:
                print("Please upload a dataset in Tab 1 first.")
                return
            
            print("Mapping Confirmed Successfully!\n")
            print(f"Identified {len(state['channels'])} unique channel touchpoints:")
            print(", ".join([f"'{c}'" for c in state['channels']]))

    lock_btn.on_click(on_lock_clicked)

    tab2_layout = widgets.VBox([
        widgets.HTML("<h4>Step 2: Map Variables & Channel Costs</h4>"),
        widgets.HBox([seq_col_w, conv_col_w, delim_w]),
        make_hr(),
        cost_box,
        widgets.HBox([lock_btn]),
        make_hr(),
        out_tab2
    ])

    # =========================================================
    # TAB 3: MODEL RESULTS
    # =========================================================
    calc_btn = widgets.Button(
        description='Compute Attribution Models', 
        button_style='success', 
        icon='chart-line',
        layout=widgets.Layout(width='auto', min_width='280px')
    )
    out_tab3 = widgets.Output()

    def parse_attribution(df, seq_col, conv_col, delim):
        converters = df[df[conv_col].astype(str).str.strip().str.lower().isin(['1', 'yes', 'true'])].copy()
        
        models = {'First_Click': {}, 'Last_Click': {}, 'Linear': {}, 'Position_Based': {}}
        
        for seq in converters[seq_col].dropna():
            touches = [t.strip() for t in str(seq).split(delim) if t.strip()]
            n_touches = len(touches)
            if n_touches == 0:
                continue
            
            models['First_Click'][touches[0]] = models['First_Click'].get(touches[0], 0) + 1.0
            models['Last_Click'][touches[-1]] = models['Last_Click'].get(touches[-1], 0) + 1.0
            
            linear_credit = 1.0 / n_touches
            for t in touches:
                models['Linear'][t] = models['Linear'].get(t, 0) + linear_credit
                
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
        with out_tab3:
            out_tab3.clear_output()
            df = state['df']
            if df is None:
                print("Please upload a CSV file in Tab 1 first.")
                return
            
            s_col = seq_col_w.value
            c_col = conv_col_w.value
            delim = delim_w.value or ','
            
            attr_summary = parse_attribution(df, s_col, c_col, delim)
            costs = {ch: state['cost_widgets'][ch].value for ch in state['channels']}
            attr_summary['Unit_Cost_$'] = attr_summary['Channel'].map(costs)
            
            print("=== ATTRIBUTED CONVERSIONS BY MODEL ===")
            display(attr_summary)
            
            fig, ax = plt.subplots(figsize=(9, 4.5))
            melted = attr_summary.melt(
                id_vars=['Channel', 'Unit_Cost_$'], 
                value_vars=['First_Click', 'Last_Click', 'Linear', 'Position_Based'],
                var_name='Model', 
                value_name='Attributed_Conversions'
            )
            
            sns.barplot(data=melted, x='Channel', y='Attributed_Conversions', hue='Model', palette='viridis', ax=ax)
            ax.set_title("Multi-Channel Attribution Comparison", fontsize=14, fontweight='bold')
            ax.set_ylabel("Attributed Conversions", fontweight='bold')
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.show()

    calc_btn.on_click(on_calc_clicked)

    tab3_layout = widgets.VBox([
        widgets.HTML("<h4>Step 3: Execute Attribution Engine & Compare Models</h4>"),
        calc_btn,
        make_hr(),
        out_tab3
    ])

    tabs = widgets.Tab(children=[tab1_layout, tab2_layout, tab3_layout])
    tabs.set_title(0, '1. Data & Preview')
    tabs.set_title(1, '2. Sequence Config & Unit Costs')
    tabs.set_title(2, '3. Model Results')
    
    display(HTML("<h3>Generic Multi-Channel Attribution Engine</h3>"))
    display(tabs)

run_generic_attribution_app()
