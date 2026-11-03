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
        
        # FIX: Use widgets.HTML instead of IPython's HTML
        widget_list = [widgets.HTML("<b>Optional: Enter Unit Cost ($ per Acquisition / Touch) for Discovered Channels:</b>")]
        
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
