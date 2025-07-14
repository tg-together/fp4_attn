import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Global storage for collected data (layer-wise)
collected_data = {}
collected_diff_data = {}
collected_mse_data = {}

# Only plot for these specific layers
PLOT_LAYERS = [2, 15, 30]

def init_layer_data(layer_idx):
    """Initialize data structures for a new layer."""
    global collected_data, collected_diff_data, collected_mse_data
    
    if layer_idx not in collected_data:
        collected_data[layer_idx] = {'q': [], 'k': [], 'v': [], 'token_counts': []}
    if layer_idx not in collected_diff_data:
        collected_diff_data[layer_idx] = {'q_diff': [], 'k_diff': [], 'v_diff': [], 'token_counts': []}
    if layer_idx not in collected_mse_data:
        collected_mse_data[layer_idx] = {'q_mse': [], 'k_mse': [], 'v_mse': [], 'token_counts': []}

def collect_qkv(q, k, v, layer_idx):
    """Collect q, k, v tensors for analysis."""
    global collected_data
    
    # Initialize layer data if needed
    init_layer_data(layer_idx)
    
    # Calculate average magnitude across tokens (0th dimension)
    # q, k, v are [B*T, H*D] tensors
    with torch.no_grad():
        num_tokens = q.shape[0]  # B*T
        q_mag = torch.mean(torch.abs(q), dim=0)  # Average over batch*time dimension
        k_mag = torch.mean(torch.abs(k), dim=0)  # Average over batch*time dimension  
        v_mag = torch.mean(torch.abs(v), dim=0)  # Average over batch*time dimension
        
        collected_data[layer_idx]['q'].append(q_mag.cpu().float().numpy())
        collected_data[layer_idx]['k'].append(k_mag.cpu().float().numpy())
        collected_data[layer_idx]['v'].append(v_mag.cpu().float().numpy())
        collected_data[layer_idx]['token_counts'].append(num_tokens)

def collect_qkv_diff(q_orig, k_orig, v_orig, q_quant, k_quant, v_quant, layer_idx):
    """Collect difference between original and quantized q, k, v tensors for analysis."""
    global collected_diff_data, collected_mse_data
    
    # Initialize layer data if needed
    init_layer_data(layer_idx)
    
    # Calculate differences and average magnitude across tokens (0th dimension)
    # All tensors are [B*T, H*D]
    with torch.no_grad():
        num_tokens = q_orig.shape[0]  # B*T
        
        # Calculate fractional differences (relative to original values)
        eps = 1e-8  # Small epsilon to avoid division by zero
        q_diff = torch.abs(q_orig - q_quant) / (torch.abs(q_orig) + eps)
        k_diff = torch.abs(k_orig - k_quant) / (torch.abs(k_orig) + eps)
        v_diff = torch.abs(v_orig - v_quant) / (torch.abs(v_orig) + eps)
        
        # Average across tokens (0th dimension)
        q_diff_mag = torch.mean(q_diff, dim=0)
        k_diff_mag = torch.mean(k_diff, dim=0)
        v_diff_mag = torch.mean(v_diff, dim=0)
        
        collected_diff_data[layer_idx]['q_diff'].append(q_diff_mag.cpu().float().numpy())
        collected_diff_data[layer_idx]['k_diff'].append(k_diff_mag.cpu().float().numpy())
        collected_diff_data[layer_idx]['v_diff'].append(v_diff_mag.cpu().float().numpy())
        collected_diff_data[layer_idx]['token_counts'].append(num_tokens)
        
        # Calculate Mean Squared Error
        q_mse = torch.mean((q_orig - q_quant)**2, dim=0)
        k_mse = torch.mean((k_orig - k_quant)**2, dim=0)
        v_mse = torch.mean((v_orig - v_quant)**2, dim=0)
        
        collected_mse_data[layer_idx]['q_mse'].append(q_mse.cpu().float().numpy())
        collected_mse_data[layer_idx]['k_mse'].append(k_mse.cpu().float().numpy())
        collected_mse_data[layer_idx]['v_mse'].append(v_mse.cpu().float().numpy())
        collected_mse_data[layer_idx]['token_counts'].append(num_tokens)

def plot_qkv_heatmaps(q, k, v, layer_idx, tag=""):
    """
    Plot 2D heatmaps of Q, K, V tensors with color legends.
    
    Args:
        q: Query tensor [B*T, H*D] 
        k: Key tensor [B*T, H*D]
        v: Value tensor [B*T, H*D]
        layer_idx: Layer index for filename and title
        tag: Tag for filename and title
        max_tokens: Maximum number of tokens to display (for readability)
        max_features: Maximum number of features to display (for readability)
    """
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Convert to numpy and move to CPU if needed
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    if isinstance(k, torch.Tensor):
        k = k.detach().cpu().numpy()
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    
    
    # Prepare titles and filenames
    title_suffix = f" (Layer {layer_idx}"
    if tag:
        title_suffix += f", {tag}"
    title_suffix += ")"
    
    file_suffix = f"_layer{layer_idx}"
    if tag:
        file_suffix += f"_{tag}"
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Q heatmap
    im1 = axes[0].imshow(q, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[0].set_title(f'Q Heatmap{title_suffix}')
    axes[0].set_xlabel('Feature Dimension')
    axes[0].set_ylabel('Token Index')
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Value')
    
    # Add statistics text
    q_min, q_max, q_mean = np.min(q), np.max(q), np.mean(q)
    axes[0].text(0.02, 0.98, f'Min: {q_min:.4f}\nMax: {q_max:.4f}\nMean: {q_mean:.4f}', 
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot K heatmap
    im2 = axes[1].imshow(k, cmap='plasma', aspect='auto', interpolation='nearest')
    axes[1].set_title(f'K Heatmap{title_suffix}')
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Token Index')
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Value')
    
    # Add statistics text
    k_min, k_max, k_mean = np.min(k), np.max(k), np.mean(k)
    axes[1].text(0.02, 0.98, f'Min: {k_min:.4f}\nMax: {k_max:.4f}\nMean: {k_mean:.4f}', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot V heatmap
    im3 = axes[2].imshow(v, cmap='inferno', aspect='auto', interpolation='nearest')
    axes[2].set_title(f'V Heatmap{title_suffix}')
    axes[2].set_xlabel('Feature Dimension')
    axes[2].set_ylabel('Token Index')
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Value')
    
    # Add statistics text
    v_min, v_max, v_mean = np.min(v), np.max(v), np.mean(v)
    axes[2].text(0.02, 0.98, f'Min: {v_min:.4f}\nMax: {v_max:.4f}\nMean: {v_mean:.4f}', 
                transform=axes[2].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'QKV Heatmaps{title_suffix}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    filename = f"plots/qkv_heatmaps{file_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"QKV Heatmaps saved as {filename}")
    print(f"Shapes: Q{q.shape}, K{k.shape}, V{v.shape}")

def plot_qkv_data(data_dict, titles, ylabel, filename, suptitle, box_color='wheat', precision=4):
    """
    Generic plotting function for QKV data.
    
    Args:
        data_dict: Dictionary with 'q', 'k', 'v' keys containing data arrays
        titles: List of titles for Q, K, V subplots
        ylabel: Y-axis label
        filename: Output filename
        suptitle: Main title for the figure
        box_color: Color for the statistics box
        precision: Number of decimal places for statistics
    """
    # Check if data exists and is not empty
    if (data_dict.get('q') is None or 
        (hasattr(data_dict['q'], 'size') and data_dict['q'].size == 0) or
        (hasattr(data_dict['q'], '__len__') and len(data_dict['q']) == 0)):
        print(f"No data available for {suptitle}")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each component (Q, K, V)
    components = ['q', 'k', 'v']
    for i, component in enumerate(components):
        data = data_dict[component]
        axes[i].plot(data)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Dimension')
        axes[i].set_ylabel(ylabel)
        
        # Calculate and display statistics
        data_min, data_max, data_mean = np.min(data), np.max(data), np.mean(data)
        stats_text = f'Min: {data_min:.{precision}f}\nMax: {data_max:.{precision}f}\nAvg: {data_mean:.{precision}f}'
        axes[i].text(0.02, 0.98, stats_text, 
                    transform=axes[i].transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.5))
    
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    
    # Ensure plots directory exists
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    
    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{suptitle} saved as {filename}")

def save_plots(tag=""):
    """Wrapper function to save plots for specific layers only."""
    global collected_data, collected_diff_data, collected_mse_data
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Iterate through only the specified layers
    for layer_idx in PLOT_LAYERS:
        if layer_idx not in collected_data or not collected_data[layer_idx]['q']:
            print(f"No data collected for layer {layer_idx}")
            continue
        
        # Prepare tag and layer suffix for titles and filenames
        title_suffix = f" (Layer {layer_idx}"
        if tag:
            title_suffix += f", {tag}"
        title_suffix += ")"
        
        file_suffix = f"_layer{layer_idx}"
        if tag:
            file_suffix += f"_{tag}"
        
        # Calculate weighted averages for magnitude data
        token_counts = np.array(collected_data[layer_idx]['token_counts'])
        weights = token_counts / np.sum(token_counts)
        
        q_data = np.array(collected_data[layer_idx]['q'])
        k_data = np.array(collected_data[layer_idx]['k'])
        v_data = np.array(collected_data[layer_idx]['v'])
        
        magnitude_data = {
            'q': np.average(q_data, axis=0, weights=weights),
            'k': np.average(k_data, axis=0, weights=weights),
            'v': np.average(v_data, axis=0, weights=weights)
        }
        
        # Plot magnitudes
        plot_qkv_data(
            data_dict=magnitude_data,
            titles=[f'Q_mag{title_suffix}', f'K_mag{title_suffix}', f'V_mag{title_suffix}'],
            ylabel='Average Magnitude',
            filename=f"plots/qkv_magnitudes{file_suffix}.png",
            suptitle=f'QKV Magnitudes{title_suffix}',
            box_color='wheat',
            precision=4
        )
        
        # Plot differences if available
        if layer_idx in collected_diff_data and collected_diff_data[layer_idx]['q_diff']:
            diff_token_counts = np.array(collected_diff_data[layer_idx]['token_counts'])
            diff_weights = diff_token_counts / np.sum(diff_token_counts)
            
            q_diff_data = np.array(collected_diff_data[layer_idx]['q_diff'])
            k_diff_data = np.array(collected_diff_data[layer_idx]['k_diff'])
            v_diff_data = np.array(collected_diff_data[layer_idx]['v_diff'])
            
            diff_data = {
                'q': np.average(q_diff_data, axis=0, weights=diff_weights),
                'k': np.average(k_diff_data, axis=0, weights=diff_weights),
                'v': np.average(v_diff_data, axis=0, weights=diff_weights)
            }
            
            plot_qkv_data(
                data_dict=diff_data,
                titles=[f'Q_diff{title_suffix}', f'K_diff{title_suffix}', f'V_diff{title_suffix}'],
                ylabel='Fractional Difference',
                filename=f"plots/qkv_diff{file_suffix}.png",
                suptitle=f'QKV Differences{title_suffix}',
                box_color='lightblue',
                precision=4
            )
        
        # Plot MSE if available
        if layer_idx in collected_mse_data and collected_mse_data[layer_idx]['q_mse']:
            mse_token_counts = np.array(collected_mse_data[layer_idx]['token_counts'])
            mse_weights = mse_token_counts / np.sum(mse_token_counts)
            
            q_mse_data = np.array(collected_mse_data[layer_idx]['q_mse'])
            k_mse_data = np.array(collected_mse_data[layer_idx]['k_mse'])
            v_mse_data = np.array(collected_mse_data[layer_idx]['v_mse'])
            
            mse_data = {
                'q': np.average(q_mse_data, axis=0, weights=mse_weights),
                'k': np.average(k_mse_data, axis=0, weights=mse_weights),
                'v': np.average(v_mse_data, axis=0, weights=mse_weights)
            }
            
            plot_qkv_data(
                data_dict=mse_data,
                titles=[f'Q_MSE{title_suffix}', f'K_MSE{title_suffix}', f'V_MSE{title_suffix}'],
                ylabel='Mean Squared Error',
                filename=f"plots/qkv_mse{file_suffix}.png",
                suptitle=f'QKV Mean Squared Error{title_suffix}',
                box_color='lightgreen',
                precision=6
            )
    
    # Clear collected data
    collected_data = {}
    collected_diff_data = {}
    collected_mse_data = {}

def plot_custom_qkv(data_dict, layer_idx, tag="", ylabel="Value", box_color='wheat', precision=4):
    """
    Custom plotting function for external use with layer support.
    
    Args:
        data_dict: Dictionary with 'q', 'k', 'v' keys containing data arrays
        layer_idx: Layer index for filename and title
        tag: Tag for filename and title
        ylabel: Y-axis label
        box_color: Color for the statistics box
        precision: Number of decimal places for statistics
    """
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    title_suffix = f" (Layer {layer_idx}"
    if tag:
        title_suffix += f", {tag}"
    title_suffix += ")"
    
    file_suffix = f"_layer{layer_idx}"
    if tag:
        file_suffix += f"_{tag}"
    
    plot_qkv_data(
        data_dict=data_dict,
        titles=[f'Q{title_suffix}', f'K{title_suffix}', f'V{title_suffix}'],
        ylabel=ylabel,
        filename=f"plots/qkv_custom{file_suffix}.png",
        suptitle=f'QKV Analysis{title_suffix}',
        box_color=box_color,
        precision=precision
    )

def get_collected_layers():
    """Return list of layers that have collected data."""
    return list(collected_data.keys())

def get_plot_layers():
    """Return list of layers that will be plotted."""
    return PLOT_LAYERS.copy()

def set_plot_layers(layers):
    """Set which layers to plot."""
    global PLOT_LAYERS
    PLOT_LAYERS = layers 