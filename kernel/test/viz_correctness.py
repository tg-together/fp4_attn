import torch
import os
import numpy as np

# Default results directory (can be overridden by function parameter)
DEFAULT_RESULTS_DIR = "/home/austin/results"

 # Number of queries per row (if AGGREGATE_HIDDEN is True)
 # or hidden dimensions per row (if AGGREGATE_HIDDEN is False)
TILE_SIZE = 128 
SHOW_HEADS = True
AGGREGATE_HIDDEN = os.getenv('AGGREGATE_HIDDEN', '0') == '1'

def assert_correctness(
    name,
    out,
    ref,
    results_dir=None,
    atol=2e-2,
    rtol=2e-2,
    verbose=False,
    assert_close=True,
    test_metadata=None,
):
    """
    out: [b, h, q, c]
    ref: [b, h, q, c]
    atol: absolute tolerance
    rtol: relative tolerance

    verbose: 
      - print max/min/median/mean/std of out, ref, abs(out - ref)
      - print percentage of large diffs (abs(out - ref) > atol
      - save html viz of large diff indices (y-axis: head, x-axis: query, large diffs are red)
        - save inputs also
        - hover over a cell to see the value of out, ref, abs(out - ref), abs(out - ref) / ref
    """
    
    # Check for NaN values first
    out_has_nan = torch.isnan(out).any()
    ref_has_nan = torch.isnan(ref).any()
    
    if out_has_nan or ref_has_nan:
        print(f"🚨 NaN DETECTED in {name}:")
        if out_has_nan:
            out_nan_count = torch.isnan(out).sum().item()
            print(f"  Output has {out_nan_count}/{out.numel()} NaN values ({out_nan_count/out.numel()*100:.2f}%)")
        if ref_has_nan:
            ref_nan_count = torch.isnan(ref).sum().item()
            print(f"  Reference has {ref_nan_count}/{ref.numel()} NaN values ({ref_nan_count/ref.numel()*100:.2f}%)")
        
        if assert_close:
            assert False, f"NaN values detected in {name}!"
        return
    
    if verbose:
        # Calculate differences
        abs_diff = torch.abs(out - ref)
        rel_diff = 2 * abs_diff / (torch.abs(ref) + torch.abs(out) + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Print statistics
        print(f"=== {name} CORRECTNESS STATISTICS ===")
        print(f"Out tensor stats:")
        print(f"  Max: {out.max().item():.6f}, Min: {out.min().item():.6f}")
        print(f"  Median: {out.median().item():.6f}, Mean: {out.mean().item():.6f}")
        print(f"  Std: {out.std().item():.6f}")
        
        print(f"Ref tensor stats:")
        print(f"  Max: {ref.max().item():.6f}, Min: {ref.min().item():.6f}")
        print(f"  Median: {ref.median().item():.6f}, Mean: {ref.mean().item():.6f}")
        print(f"  Std: {ref.std().item():.6f}")
        
        print(f"Absolute difference stats:")
        print(f"  Max: {abs_diff.max().item():.6f}, Min: {abs_diff.min().item():.6f}")
        print(f"  Median: {abs_diff.median().item():.6f}, Mean: {abs_diff.mean().item():.6f}")
        print(f"  Std: {abs_diff.std().item():.6f}")

        print(f"Relative difference stats:")
        print(f"  Max: {rel_diff.max().item():.6f}, Min: {rel_diff.min().item():.6f}")
        print(f"  Median: {rel_diff.median().item():.6f}, Mean: {rel_diff.mean().item():.6f}")
        print(f"  Std: {rel_diff.std().item():.6f}")
        
        # Calculate percentage of large diffs
        large_diffs = abs_diff > atol
        large_rel_diffs = rel_diff > rtol
        # combined_large_diffs = large_diffs
        combined_large_diffs = large_diffs | large_rel_diffs
        
        total_elements = out.numel()
        large_diff_count = combined_large_diffs.sum().item()
        large_diff_percentage = (large_diff_count / total_elements) * 100
        
        print(f"Large differences (abs > {atol} OR rel > {rtol}):")
        print(f"  Count: {large_diff_count}/{total_elements}")
        print(f"  Percentage: {large_diff_percentage:.2f}%")
        
        # Generate HTML visualization if there are large diffs
        # _generate_html_visualization(name, out, ref, abs_diff, rel_diff, combined_large_diffs, atol, rtol, results_dir)
    
    if assert_close:
        try:
            assert torch.allclose(out, ref, atol=atol, rtol=rtol)
        except Exception as e:
            _generate_html_visualization(name, out, ref, abs_diff, rel_diff, combined_large_diffs, atol, rtol, results_dir)
            print(f"🚨 Assertion failed in {name}:")
            print(f"  {e}")
            assert False, f"Assertion failed in {name}!"


def _generate_html_visualization(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir=None):
    """Generate HTML visualization of large differences."""
    b, h, q, c = out.shape
    
    # Use provided results_dir or fall back to default
    actual_results_dir = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    
    # Save input tensors
    os.makedirs(actual_results_dir, exist_ok=True)
    # torch.save(out, f"{RESULTS_DIR}/out_tensor.pt")
    # torch.save(ref, f"{RESULTS_DIR}/ref_tensor.pt")
    # torch.save(abs_diff, f"{RESULTS_DIR}/abs_diff.pt")
    # torch.save(rel_diff, f"{RESULTS_DIR}/rel_diff.pt")
    
    # Flexible visualization based on flags:
    # SHOW_HEADS=True, AGGREGATE_HIDDEN=False: Each head separately with full hidden dims
    # SHOW_HEADS=True, AGGREGATE_HIDDEN=True: Each head separately, aggregate channels
    # SHOW_HEADS=False, AGGREGATE_HIDDEN=False: Aggregate heads, show full hidden dims  
    # SHOW_HEADS=False, AGGREGATE_HIDDEN=True: Aggregate both heads and channels
    
    if SHOW_HEADS and not AGGREGATE_HIDDEN:
        # Show each head separately with full hidden dimension detail
        _generate_per_head_per_hidden_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir)
    elif not SHOW_HEADS and not AGGREGATE_HIDDEN:
        # Aggregate across heads, show full hidden dimensions (original per-hidden view)
        _generate_per_position_per_hidden_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir)
    elif SHOW_HEADS and AGGREGATE_HIDDEN:
        # Show each head separately, aggregate across channels (original per-head view)
        _generate_per_head_aggregated_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir)
    else:  # not SHOW_HEADS and AGGREGATE_HIDDEN
        # Aggregate across both heads and channels (most compact view)
        _generate_fully_aggregated_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir)


def _generate_per_head_per_hidden_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir):
    """Generate visualization showing each head separately with full hidden dimension detail."""
    b, h, q, c = out.shape
    actual_results_dir = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    
    for batch_idx in range(b):
        # Slice batch specific tensors
        out_np = out[batch_idx].to(torch.float32).detach().cpu().numpy()       # [h, q, c]
        ref_np = ref[batch_idx].to(torch.float32).detach().cpu().numpy()
        abs_diff_np = abs_diff[batch_idx].to(torch.float32).detach().cpu().numpy()
        rel_diff_np = rel_diff[batch_idx].to(torch.float32).detach().cpu().numpy()
        large_diffs_np = large_diffs[batch_idx].to(torch.float32).detach().cpu().numpy()

        # Layout parameters
        dims_per_row = min(TILE_SIZE, c)
        num_rows_hidden = (c + dims_per_row - 1) // dims_per_row
        available_width = 1300
        optimal_cell_size = min(15, max(4, available_width // dims_per_row))

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Attention Correctness (Per-Head Per-Hidden) - Batch {batch_idx}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; overflow-x: auto; }}
        .container {{ min-width: fit-content; }}
        .head-section {{ margin-bottom: 30px; border: 2px solid #333; border-radius: 8px; padding: 15px; }}
        .head-title {{ font-weight: bold; font-size: 18px; margin-bottom: 15px; color: #333; }}
        .position-section {{ margin-bottom: 20px; border: 1px solid #ccc; border-radius: 4px; padding: 10px; }}
        .position-title {{ font-weight: bold; margin-bottom: 10px; color: #666; }}
        .hidden-row {{ display: grid; grid-template-columns: repeat({dims_per_row}, {optimal_cell_size}px); gap: 1px; margin-bottom: 2px; }}
        .row-label {{ font-size: 10px; color: #666; margin-bottom: 2px; }}
        .cell {{ width: {optimal_cell_size}px; height: {optimal_cell_size}px; border: 1px solid #ccc; cursor: pointer; }}
        .cell.large-diff {{ background-color: #ff4444; }}
        .cell.small-diff {{ background-color: #44ff44; }}
        .tooltip {{ position: fixed; background-color: #333; color: white; padding: 8px; border-radius: 4px; font-size: 12px; z-index: 1000; display: none; white-space: pre-line; }}
        .info {{ margin: 10px 0; background-color: #f5f5f5; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Per-Head Per-Hidden Visualization - Batch {batch_idx}</h1>
        <div class="info">
            <strong>Shape:</strong> [{b}, {h}, {q}, {c}] | <strong>Tolerances:</strong> abs={atol}, rel={rtol}<br>
            <strong>Legend:</strong> <span style="background-color: #ff4444; color: white; padding: 2px;">Red = Large Diff</span> 
            <span style="background-color: #44ff44; color: black; padding: 2px;">Green = OK</span>
        </div>
"""

        # Generate section for each head
        for head_idx in range(h):
            # Calculate head-level statistics across all positions and channels
            head_abs_diff = abs_diff_np[head_idx]  # [q, c]
            head_rel_diff = rel_diff_np[head_idx]  # [q, c]
            
            head_abs_max = float(head_abs_diff.max())
            head_abs_median = float(np.median(head_abs_diff))
            head_abs_mean = float(head_abs_diff.mean())
            
            head_rel_max = float(head_rel_diff.max())
            head_rel_median = float(np.median(head_rel_diff))
            head_rel_mean = float(head_rel_diff.mean())
            
            html_content += f'<div class="head-section"><div class="head-title">Head {head_idx}<br>Abs (max: {head_abs_max:.4f}, med: {head_abs_median:.4f}, mean: {head_abs_mean:.4f})<br>Rel (max: {head_rel_max:.4f}, med: {head_rel_median:.4f}, mean: {head_rel_mean:.4f})</div>'
            
            # Generate section for each position within this head
            for pos_idx in range(q):
                # Calculate position-level statistics across all channels for this head
                pos_abs_diff = abs_diff_np[head_idx, pos_idx]  # [c]
                pos_rel_diff = rel_diff_np[head_idx, pos_idx]  # [c]
                
                pos_abs_max = float(pos_abs_diff.max())
                pos_abs_median = float(np.median(pos_abs_diff))
                pos_abs_mean = float(pos_abs_diff.mean())
                
                pos_rel_max = float(pos_rel_diff.max())
                pos_rel_median = float(np.median(pos_rel_diff))
                pos_rel_mean = float(pos_rel_diff.mean())
                
                html_content += f'<div class="position-section"><div class="position-title">Position {pos_idx}<br>Abs (max: {pos_abs_max:.4f}, med: {pos_abs_median:.4f}, mean: {pos_abs_mean:.4f})<br>Rel (max: {pos_rel_max:.4f}, med: {pos_rel_median:.4f}, mean: {pos_rel_mean:.4f})</div>'
                
                # Generate rows of hidden dimensions
                for row_idx in range(num_rows_hidden):
                    start_chan = row_idx * dims_per_row
                    end_chan = min(start_chan + dims_per_row, c)
                    
                    html_content += f'<div class="row-label">Hidden dims {start_chan}-{end_chan-1}</div><div class="hidden-row">'
                    
                    for col_idx in range(dims_per_row):
                        chan_idx = start_chan + col_idx
                        
                        if chan_idx < c:
                            is_large_diff = bool(large_diffs_np[head_idx, pos_idx, chan_idx])
                            out_val = float(out_np[head_idx, pos_idx, chan_idx])
                            ref_val = float(ref_np[head_idx, pos_idx, chan_idx])
                            abs_val = float(abs_diff_np[head_idx, pos_idx, chan_idx])
                            rel_val = float(rel_diff_np[head_idx, pos_idx, chan_idx])
                            
                            tooltip_text = f"Head: {head_idx}, Pos: {pos_idx}, Chan: {chan_idx}\\nOut: {out_val:.6f}\\nRef: {ref_val:.6f}\\nAbs: {abs_val:.6f}\\nRel: {rel_val:.6f}"
                            cell_class = "large-diff" if is_large_diff else "small-diff"
                            
                            html_content += f'<div class="cell {cell_class}" onmouseover="showTooltip(event, \'{tooltip_text}\')" onmouseout="hideTooltip()"></div>'
                        else:
                            html_content += '<div class="cell" style="background-color: #f0f0f0;"></div>'
                    
                    html_content += '</div>'
                
                html_content += '</div>'  # Close position section
            
            html_content += '</div>'  # Close head section

        html_content += '''
        <div class="tooltip" id="tooltip"></div>
    </div>
    <script>
        function showTooltip(event, text) {
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = text;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY + 10) + 'px';
        }
        function hideTooltip() {
            document.getElementById('tooltip').style.display = 'none';
        }
    </script>
</body>
</html>'''

        html_filename = f"{actual_results_dir}/{name}_batch_{batch_idx}_per_head_per_hidden.html"
        with open(html_filename, 'w') as f:
            f.write(html_content)
        print(f"Saved HTML visualization: {html_filename}")


def _generate_per_position_per_hidden_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir):
    """Aggregate across heads, show full hidden dimensions (original per-hidden view)."""
    b, h, q, c = out.shape
    actual_results_dir = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    
    for batch_idx in range(b):
        # Aggregate across heads - show position x hidden dimension view
        batch_abs_diff = abs_diff[batch_idx]         # [h, q, c]
        batch_large_diffs = large_diffs[batch_idx]   # [h, q, c]

        out_np = out[batch_idx].to(torch.float32).detach().cpu().numpy()       # [h, q, c]
        ref_np = ref[batch_idx].to(torch.float32).detach().cpu().numpy()
        abs_diff_np = batch_abs_diff.to(torch.float32).detach().cpu().numpy()
        rel_diff_np = rel_diff[batch_idx].to(torch.float32).detach().cpu().numpy()

        # Layout parameters for hidden-dimension cells
        dims_per_row = min(TILE_SIZE, c)
        num_rows_hidden = (c + dims_per_row - 1) // dims_per_row

        available_width = 1300
        optimal_cell_size = min(20, max(6, available_width // dims_per_row))

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Attention Correctness Visualization (Per-Position Per-Hidden) - Batch {batch_idx}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            overflow-x: auto;
        }}
        .container {{
            min-width: fit-content;
        }}
        .grid-container {{
            overflow-x: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }}
        .position-section {{
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
        }}
        .position-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .hidden-row {{
            display: grid;
            grid-template-columns: repeat({dims_per_row}, {optimal_cell_size}px);
            gap: 1px;
            margin-bottom: 2px;
            width: fit-content;
        }}
        .row-label {{
            font-size: 10px;
            color: #666;
            margin-bottom: 2px;
        }}
        .cell {{
            width: {optimal_cell_size}px;
            height: {optimal_cell_size}px;
            border: 1px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: {max(6, optimal_cell_size//4)}px;
            cursor: pointer;
            position: relative;
            min-width: {optimal_cell_size}px;
        }}
        .cell.large-diff {{
            background-color: #ff4444;
            color: white;
        }}
        .cell.small-diff {{
            background-color: #44ff44;
            color: black;
        }}
        .tooltip {{
            position: fixed;
            background-color: #333;
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
            white-space: pre-line;
            max-width: 350px;
            display: none;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .info {{
            margin: 10px 0;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
        }}
        .axis-label {{
            font-weight: bold;
            margin: 10px 0;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Per-Position Per-Hidden Visualization - Batch {batch_idx}</h1>
        <div class="info">
            <strong>Shape:</strong> [{b}, {h}, {q}, {c}] | <strong>Tolerances:</strong> abs={atol}, rel={rtol}<br>
            <strong>Legend:</strong> <span style="background-color: #ff4444; color: white; padding: 2px;">Red = Large Diff</span> 
            <span style="background-color: #44ff44; color: black; padding: 2px;">Green = OK</span><br>
            <strong>View:</strong> Aggregated across all heads, showing positions x hidden dimensions
        </div>
        <div class="grid-container">
"""

        # Generate a section for every position (query index)
        for pos_idx in range(q):
            # Calculate position-level statistics across all heads and channels
            pos_abs_diff_all_heads = abs_diff_np[:, pos_idx, :]  # [h, c]
            pos_rel_diff_all_heads = rel_diff_np[:, pos_idx, :]  # [h, c]
            
            pos_abs_max = float(pos_abs_diff_all_heads.max())
            pos_abs_median = float(np.median(pos_abs_diff_all_heads))
            pos_abs_mean = float(pos_abs_diff_all_heads.mean())
            
            pos_rel_max = float(pos_rel_diff_all_heads.max())
            pos_rel_median = float(np.median(pos_rel_diff_all_heads))
            pos_rel_mean = float(pos_rel_diff_all_heads.mean())
            
            html_content += f"""
            <div class="position-section">
                <div class="position-title">Position {pos_idx}<br>Abs (max: {pos_abs_max:.4f}, med: {pos_abs_median:.4f}, mean: {pos_abs_mean:.4f})<br>Rel (max: {pos_rel_max:.4f}, med: {pos_rel_median:.4f}, mean: {pos_rel_mean:.4f})</div>
"""

            # Pre-compute per-channel aggregates across heads for this position
            pos_abs_diff_tensor = batch_abs_diff[:, pos_idx, :]            # torch [h,c]
            large_diff_tensor    = batch_large_diffs[:, pos_idx, :]

            max_abs_diff_per_chan = pos_abs_diff_tensor.max(dim=0)[0].to(torch.float32).cpu().numpy()  # [c]
            has_large_diff_chan  = large_diff_tensor.any(dim=0).cpu().numpy()                           # [c]

            pos_abs_diff_np = pos_abs_diff_tensor.to(torch.float32).cpu().numpy()  # [h, c]

            # Row wise hidden-channel grid
            for row_idx in range(num_rows_hidden):
                start_chan = row_idx * dims_per_row
                end_chan   = min(start_chan + dims_per_row, c)

                html_content += f"""
                <div class="row-label">Hidden dims {start_chan}-{end_chan-1}</div>
                <div class="hidden-row">
"""

                for col_idx in range(dims_per_row):
                    chan_idx = start_chan + col_idx

                    if chan_idx < c:
                        is_large_diff = bool(has_large_diff_chan[chan_idx])
                        max_diff      = float(max_abs_diff_per_chan[chan_idx])

                        # Identify the head with the max diff for detailed tooltip
                        head_idx_max = int(np.argmax(pos_abs_diff_np[:, chan_idx]))

                        out_val  = float(out_np[head_idx_max, pos_idx, chan_idx])
                        ref_val  = float(ref_np[head_idx_max, pos_idx, chan_idx])
                        abs_val  = float(abs_diff_np[head_idx_max, pos_idx, chan_idx])
                        rel_val  = float(rel_diff_np[head_idx_max, pos_idx, chan_idx])

                        tooltip_text = (
                            f"Position: {pos_idx}, Hidden Dim: {chan_idx}\\n"
                            f"Max abs diff: {max_diff:.6f} (head {head_idx_max})\\n"
                            f"Out: {out_val:.6f}\\n"
                            f"Ref: {ref_val:.6f}\\n"
                            f"Abs diff: {abs_val:.6f}\\n"
                            f"Rel diff: {rel_val:.6f}"
                            )

                        cell_class = "large-diff" if is_large_diff else "small-diff"

                        html_content += (
                            f"<div class=\"cell {cell_class}\" "
                            f"onmouseover=\"showTooltip(event, '{tooltip_text}')\" "
                            f"onmouseout=\"hideTooltip()\"></div>"
                        )
                    else:
                        html_content += "<div class=\"cell\" style=\"background-color: #f0f0f0; border: 1px solid #ddd;\"></div>"

                html_content += """
                </div>
"""

            html_content += """
            </div>
"""

        html_content += f"""
        </div>
        <div class="tooltip" id="tooltip"></div>
    </div>
    <script>
        function showTooltip(event, text) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = text;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY + 10) + 'px';
        }}
        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}
    </script>
</body>
</html>
"""

        html_filename = f"{actual_results_dir}/{name}_batch_{batch_idx}_per_position_per_hidden.html"
        with open(html_filename, 'w') as f:
            f.write(html_content)
        print(f"Saved HTML visualization: {html_filename}")


def _generate_per_head_aggregated_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir):
    """Show each head separately, aggregate across channels (original per-head view)."""
    b, h, q, c = out.shape
    actual_results_dir = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    
    for batch_idx in range(b):
        # Aggregate large diffs across the channel dimension for visualization
        batch_abs_diff = abs_diff[batch_idx]  # Shape: [h, q, c]
        batch_large_diffs = large_diffs[batch_idx]  # Shape: [h, q, c]
        
        # Get max absolute difference per (head, query) position
        max_abs_diff_per_pos = batch_abs_diff.max(dim=2)[0]  # Shape: [h, q]
        has_large_diff_per_pos = batch_large_diffs.any(dim=2)  # Shape: [h, q]
        
        # Convert to numpy for easier manipulation
        max_abs_diff_np = max_abs_diff_per_pos.to(torch.float32).detach().cpu().numpy()
        has_large_diff_np = has_large_diff_per_pos.to(torch.float32).detach().cpu().numpy()
        out_np = out[batch_idx].to(torch.float32).detach().cpu().numpy()
        ref_np = ref[batch_idx].to(torch.float32).detach().cpu().numpy()
        abs_diff_np = batch_abs_diff.to(torch.float32).detach().cpu().numpy()
        rel_diff_np = rel_diff[batch_idx].to(torch.float32).detach().cpu().numpy()
        
        # Calculate tiling parameters
        queries_per_row = min(TILE_SIZE, q)
        num_rows = (q + queries_per_row - 1) // queries_per_row  # Ceiling division
        
        # Calculate optimal cell size based on available width
        available_width = 1300
        optimal_cell_size = min(20, max(6, available_width // queries_per_row))
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Attention Correctness Visualization (Per-Head Aggregated) - Batch {batch_idx}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            overflow-x: auto;
        }}
        .container {{
            min-width: fit-content;
        }}
        .grid-container {{
            overflow-x: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }}
        .head-section {{
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
        }}
        .head-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .query-row {{
            display: grid;
            grid-template-columns: repeat({queries_per_row}, {optimal_cell_size}px);
            gap: 1px;
            margin-bottom: 2px;
            width: fit-content;
        }}
        .row-label {{
            font-size: 10px;
            color: #666;
            margin-bottom: 2px;
        }}
        .cell {{
            width: {optimal_cell_size}px;
            height: {optimal_cell_size}px;
            border: 1px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: {max(6, optimal_cell_size//4)}px;
            cursor: pointer;
            position: relative;
            min-width: {optimal_cell_size}px;
        }}
        .cell.large-diff {{
            background-color: #ff4444;
            color: white;
        }}
        .cell.small-diff {{
            background-color: #44ff44;
            color: black;
        }}
        .tooltip {{
            position: fixed;
            background-color: #333;
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
            white-space: pre-line;
            max-width: 350px;
            display: none;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .info {{
            margin: 10px 0;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
        }}
        .axis-label {{
            font-weight: bold;
            margin: 10px 0;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Per-Head Aggregated Visualization - Batch {batch_idx}</h1>
        <div class="info">
            <strong>Shape:</strong> [{b}, {h}, {q}, {c}] | <strong>Tolerances:</strong> abs={atol}, rel={rtol}<br>
            <strong>Legend:</strong> <span style="background-color: #ff4444; color: white; padding: 2px;">Red = Large Diff</span> 
            <span style="background-color: #44ff44; color: black; padding: 2px;">Green = OK</span><br>
            <strong>View:</strong> Each head separately, aggregated across channels
        </div>
        <div class="grid-container">
"""
        
        # Generate sections for each head
        for head_idx in range(h):
            # Calculate head-level statistics across all queries and channels
            head_abs_diff = abs_diff_np[head_idx]  # [q, c]
            head_rel_diff = rel_diff_np[head_idx]  # [q, c]
            
            head_abs_max = float(head_abs_diff.max())
            head_abs_median = float(np.median(head_abs_diff))
            head_abs_mean = float(head_abs_diff.mean())
            
            head_rel_max = float(head_rel_diff.max())
            head_rel_median = float(np.median(head_rel_diff))
            head_rel_mean = float(head_rel_diff.mean())
            
            html_content += f"""
            <div class="head-section">
                <div class="head-title">Head {head_idx}<br>Abs (max: {head_abs_max:.4f}, med: {head_abs_median:.4f}, mean: {head_abs_mean:.4f})<br>Rel (max: {head_rel_max:.4f}, med: {head_rel_median:.4f}, mean: {head_rel_mean:.4f})</div>
"""
            
            # Generate rows of queries for this head
            for row_idx in range(num_rows):
                start_query = row_idx * queries_per_row
                end_query = min(start_query + queries_per_row, q)
                
                html_content += f"""
                <div class="row-label">Queries {start_query}-{end_query-1}</div>
                <div class="query-row">
"""
                
                # Generate cells for this row
                for col_idx in range(queries_per_row):
                    query_idx = start_query + col_idx
                    
                    if query_idx < q:
                        is_large_diff = has_large_diff_np[head_idx, query_idx]
                        max_diff = max_abs_diff_np[head_idx, query_idx]
                        
                        cell_class = "large-diff" if is_large_diff else "small-diff"
                        
                        # Calculate statistics for this position across all channels
                        pos_out = out_np[head_idx, query_idx]  # Shape: [c]
                        pos_ref = ref_np[head_idx, query_idx]  # Shape: [c]
                        pos_abs_diff = abs_diff_np[head_idx, query_idx]  # Shape: [c]
                        pos_rel_diff = rel_diff_np[head_idx, query_idx]  # Shape: [c]
                        
                        # Find the channel with maximum absolute difference
                        max_diff_channel = np.argmax(pos_abs_diff)
                        
                        tooltip_text = f"Head: {head_idx}, Query: {query_idx}\\n"
                        tooltip_text += f"Max abs diff: {max_diff:.6f} (channel {max_diff_channel})\\n"
                        tooltip_text += f"Out[max_ch]: {pos_out[max_diff_channel]:.6f}\\n"
                        tooltip_text += f"Ref[max_ch]: {pos_ref[max_diff_channel]:.6f}\\n"
                        tooltip_text += f"Abs diff[max_ch]: {pos_abs_diff[max_diff_channel]:.6f}\\n"
                        tooltip_text += f"Rel diff[max_ch]: {pos_rel_diff[max_diff_channel]:.6f}\\n"
                        tooltip_text += f"Avg out: {pos_out.mean():.6f}\\n"
                        tooltip_text += f"Avg ref: {pos_ref.mean():.6f}\\n"
                        tooltip_text += f"Avg abs diff: {pos_abs_diff.mean():.6f}"
                        
                        html_content += f"""
                    <div class="cell {cell_class}" 
                         onmouseover="showTooltip(event, '{tooltip_text}')"
                         onmouseout="hideTooltip()">
                    </div>
"""
                    else:
                        # Empty cell for padding
                        html_content += f"""
                    <div class="cell" style="background-color: #f0f0f0; border: 1px solid #ddd;">
                    </div>
"""
                
                html_content += """
                </div>
"""
            
            html_content += """
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="tooltip" id="tooltip"></div>
    </div>
    
    <script>
        function showTooltip(event, text) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = text;
            tooltip.style.display = 'block';
            
            let left = event.clientX + 10;
            let top = event.clientY + 10;
            
            const tooltipRect = tooltip.getBoundingClientRect();
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            
            if (left + tooltipRect.width > viewportWidth) {{
                left = event.clientX - tooltipRect.width - 10;
            }}
            if (top + tooltipRect.height > viewportHeight) {{
                top = event.clientY - tooltipRect.height - 10;
            }}
            
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        }}
        
        function hideTooltip() {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.display = 'none';
        }}
    </script>
</body>
</html>
"""
        
        html_filename = f"{actual_results_dir}/{name}_batch_{batch_idx}_per_head_aggregated.html"
        with open(html_filename, 'w') as f:
            f.write(html_content)
        
        print(f"Saved HTML visualization: {html_filename}")


def _generate_fully_aggregated_viz(name, out, ref, abs_diff, rel_diff, large_diffs, atol, rtol, results_dir):
    """Most compact view - aggregate across both heads and channels, showing only query positions."""
    b, h, q, c = out.shape
    actual_results_dir = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    
    for batch_idx in range(b):
        # Aggregate across both heads and channels - show only query positions
        batch_abs_diff = abs_diff[batch_idx]         # [h, q, c]
        batch_large_diffs = large_diffs[batch_idx]   # [h, q, c]
        
        # Aggregate across heads and channels to get per-query statistics
        max_abs_diff_per_query = batch_abs_diff.view(h*c, q).max(dim=0)[0]  # [q]
        has_large_diff_per_query = batch_large_diffs.view(h*c, q).any(dim=0)  # [q]
        
        out_np = out[batch_idx].to(torch.float32).detach().cpu().numpy()       # [h, q, c]
        ref_np = ref[batch_idx].to(torch.float32).detach().cpu().numpy()
        abs_diff_np = batch_abs_diff.to(torch.float32).detach().cpu().numpy()
        rel_diff_np = rel_diff[batch_idx].to(torch.float32).detach().cpu().numpy()
        
        max_abs_diff_np = max_abs_diff_per_query.to(torch.float32).cpu().numpy()
        has_large_diff_np = has_large_diff_per_query.cpu().numpy()
        
        # Layout parameters
        queries_per_row = min(TILE_SIZE, q)
        num_rows = (q + queries_per_row - 1) // queries_per_row
        
        available_width = 1300
        optimal_cell_size = min(20, max(6, available_width // queries_per_row))
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Attention Correctness Visualization (Fully Aggregated) - Batch {batch_idx}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            overflow-x: auto;
        }}
        .container {{
            min-width: fit-content;
        }}
        .grid-container {{
            overflow-x: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }}
        .query-row {{
            display: grid;
            grid-template-columns: repeat({queries_per_row}, {optimal_cell_size}px);
            gap: 1px;
            margin-bottom: 2px;
            width: fit-content;
        }}
        .row-label {{
            font-size: 10px;
            color: #666;
            margin-bottom: 2px;
        }}
        .cell {{
            width: {optimal_cell_size}px;
            height: {optimal_cell_size}px;
            border: 1px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: {max(6, optimal_cell_size//4)}px;
            cursor: pointer;
            position: relative;
            min-width: {optimal_cell_size}px;
        }}
        .cell.large-diff {{
            background-color: #ff4444;
            color: white;
        }}
        .cell.small-diff {{
            background-color: #44ff44;
            color: black;
        }}
        .tooltip {{
            position: fixed;
            background-color: #333;
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
            white-space: pre-line;
            max-width: 350px;
            display: none;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .info {{
            margin: 10px 0;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fully Aggregated Visualization - Batch {batch_idx}</h1>
        <div class="info">
            <strong>Shape:</strong> [{b}, {h}, {q}, {c}] | <strong>Tolerances:</strong> abs={atol}, rel={rtol}<br>
            <strong>Legend:</strong> <span style="background-color: #ff4444; color: white; padding: 2px;">Red = Large Diff</span> 
            <span style="background-color: #44ff44; color: black; padding: 2px;">Green = OK</span><br>
            <strong>View:</strong> Most compact - aggregated across both heads and channels
        </div>
        <div class="grid-container">
"""
        
        # Generate rows of queries
        for row_idx in range(num_rows):
            start_query = row_idx * queries_per_row
            end_query = min(start_query + queries_per_row, q)
            
            html_content += f"""
            <div class="row-label">Queries {start_query}-{end_query-1}</div>
            <div class="query-row">
"""
            
            for col_idx in range(queries_per_row):
                query_idx = start_query + col_idx
                
                if query_idx < q:
                    is_large_diff = bool(has_large_diff_np[query_idx])
                    max_diff = float(max_abs_diff_np[query_idx])
                    
                    cell_class = "large-diff" if is_large_diff else "small-diff"
                    
                    # Find overall max difference location across all heads and channels
                    query_abs_diff = abs_diff_np[:, query_idx, :]  # [h, c]
                    max_indices = np.unravel_index(np.argmax(query_abs_diff), query_abs_diff.shape)
                    max_head, max_chan = max_indices
                    
                    # Get statistics for the max difference location
                    out_val = float(out_np[max_head, query_idx, max_chan])
                    ref_val = float(ref_np[max_head, query_idx, max_chan])
                    abs_val = float(abs_diff_np[max_head, query_idx, max_chan])
                    rel_val = float(rel_diff_np[max_head, query_idx, max_chan])
                    
                    # Overall statistics across all heads/channels for this query
                    query_out_mean = float(out_np[:, query_idx, :].mean())
                    query_ref_mean = float(ref_np[:, query_idx, :].mean())
                    query_abs_diff_mean = float(abs_diff_np[:, query_idx, :].mean())
                    
                    tooltip_text = f"Query: {query_idx}\\n"
                    tooltip_text += f"Max abs diff: {max_diff:.6f} (head {max_head}, chan {max_chan})\\n"
                    tooltip_text += f"Out[max]: {out_val:.6f}\\n"
                    tooltip_text += f"Ref[max]: {ref_val:.6f}\\n"
                    tooltip_text += f"Abs diff[max]: {abs_val:.6f}\\n"
                    tooltip_text += f"Rel diff[max]: {rel_val:.6f}\\n"
                    tooltip_text += f"Avg out: {query_out_mean:.6f}\\n"
                    tooltip_text += f"Avg ref: {query_ref_mean:.6f}\\n"
                    tooltip_text += f"Avg abs diff: {query_abs_diff_mean:.6f}"
                    
                    html_content += f"""
                <div class="cell {cell_class}" 
                     onmouseover="showTooltip(event, '{tooltip_text}')"
                     onmouseout="hideTooltip()">
                </div>
"""
                else:
                    html_content += f"""
                <div class="cell" style="background-color: #f0f0f0; border: 1px solid #ddd;">
                </div>
"""
            
            html_content += """
            </div>
"""
        
        html_content += f"""
        </div>
        <div class="tooltip" id="tooltip"></div>
    </div>
    <script>
        function showTooltip(event, text) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = text;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY + 10) + 'px';
        }}
        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}
    </script>
</body>
</html>
"""
        
        html_filename = f"{actual_results_dir}/{name}_batch_{batch_idx}_fully_aggregated.html"
        with open(html_filename, 'w') as f:
            f.write(html_content)
        
        print(f"Saved HTML visualization: {html_filename}")