import pandas as pd 
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import os 
import numpy as np 
from typing import Optional, List, Dict
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

def plot_class_samples(df: pd.DataFrame, config_dir: str, n_classes: int = 5, max_images: int = 8):
    """
    Sample random class_ids and create plots showing sample images for each class.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: image_path, easting, northing, class_id
    config_dir : str
        Directory where the config files are saved
    n_classes : int, default 5
        Number of random classes to sample
    max_images : int, default 8
        Maximum number of images to show per class
    """
    
    # Get all unique class_ids and sample n_classes
    unique_classes = df['class_id'].unique()
    sampled_classes = random.sample(list(unique_classes), min(n_classes, len(unique_classes)))
    
    config_path_obj = Path(config_dir)
    
    for i, class_id in enumerate(sampled_classes):
        # Get all images for this class
        class_data = df[df['class_id'] == class_id]
        
        # Sample up to max_images
        n_samples = min(max_images, len(class_data))
        sampled_images = class_data.sample(n=n_samples, random_state=42)
        
        # Create subplot grid
        if n_samples <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 4
            
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
        fig.suptitle(f'Class {class_id} - Sample Images ({len(class_data)} total images)', fontsize=14)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten() if n_samples > 1 else [axes]
        
        for j, (idx, row) in enumerate(sampled_images.iterrows()):
            if j >= len(axes_flat):
                break
                
            ax = axes_flat[j]
            
            # Load and display image
            dirpath = df.attrs.get('dataset_folder', '')
            img_path = os.path.join(dirpath, row['image_path'])
            
            # Handle both absolute and relative paths
            if not Path(img_path).is_absolute():
                full_path = config_path_obj.parent / img_path
            else:
                full_path = Path(img_path)
            
            if full_path.exists():
                img = Image.open(full_path)
                ax.imshow(img)
                ax.set_title(f'E: {row["easting"]:.1f}, N: {row["northing"]:.1f}', fontsize=10)
            else:
                # If image not found, show placeholder
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'E: {row["easting"]:.1f}, N: {row["northing"]:.1f}', fontsize=10)
            
            ax.axis('off')
        
        # Hide unused subplots
        for j in range(n_samples, len(axes_flat)):
            axes_flat[j].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = config_path_obj / f'class_samples/class_{class_id}_samples.png'
        os.makedirs(plot_path.parent, exist_ok=True)
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        



def plot_class_sim_heatmap(df: pd.DataFrame, class_sims: Dict[int, np.ndarray], plot_path: str, classes: Optional[List[int]] = None): 
    import math

    if classes is None: 
        classes = np.random.choice(np.unique(df['class_id']), size=min(5, len(np.unique(df['class_id']))), replace=False)

    n_classes = len(classes)
    n_cols = min(3, n_classes)
    n_rows = math.ceil(n_classes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Compute global vmin/vmax for all selected classes, ignoring diagonal (self-similarity)
    all_offdiag_vals = []
    for class_id in classes:
        sim_matrix = class_sims[class_id]
        # Get off-diagonal values only
        offdiag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
        all_offdiag_vals.append(offdiag)
    all_offdiag_vals = np.concatenate(all_offdiag_vals)
    vmin = all_offdiag_vals.min()
    vmax = all_offdiag_vals.max()

    for i, class_id in enumerate(classes):
        sim_matrix = class_sims[class_id]
        # Compute mean/std for off-diagonal only
        offdiag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
        mean = offdiag.mean()
        std = offdiag.std()
        ax = axes[i]
        im = ax.imshow(sim_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Class {class_id} Similarity\nmean={mean:.4f}, std={std:.4f}')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(len(classes), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_intra_class_sim_dist(class_sims: Dict[int, np.ndarray], means: List[float], stds: List[float], plot_path: str):
    class_sizes = [sim_matrix.shape[0] for sim_matrix in class_sims.values()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Distribution of means
    axes[0, 0].hist(means, bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Class Cohernace \n (Higher = more Coherent)')
    axes[0, 0].set_xlabel('Mean Similarity')
    axes[0, 0].set_ylabel('Number of Classes')
    
    # 2. Distribution of stds
    axes[0, 1].hist(stds, bins=30, color='salmon', edgecolor='black')
    axes[0, 1].set_title('Distribution of Class Diversity \n (Higher = more Diverse)')
    axes[0, 1].set_xlabel('Std of Intra-class Similarity')
    axes[0, 1].set_ylabel('Number of Classes')
    
    # 3. Scatter: mean vs std
    axes[1, 0].scatter(means, stds, c='purple', alpha=0.7)
    axes[1, 0].set_title('Mean vs Std of Intra-Class Similarity')
    axes[1, 0].set_xlabel('Mean Similarity')
    axes[1, 0].set_ylabel('Std of Similarity')
    
    # 4. Scatter: std vs class size
    axes[1, 1].scatter(class_sizes, stds, c='green', alpha=0.7)
    axes[1, 1].set_title('Std of Similarity vs Class Size')
    axes[1, 1].set_xlabel('Class Size')
    axes[1, 1].set_ylabel('Std of Similarity')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_inter_class_sim_dist(inter_class_sim: np.ndarray, plot_path: str):
    """
    inter_class_sim: [C, C] similarity between class centroids (higher = more similar).
    plot_path: path to save the plot
    """
    assert inter_class_sim.ndim == 2 and inter_class_sim.shape[0] == inter_class_sim.shape[1], \
        "inter_class_sim must be a square [C, C] matrix"
    C = inter_class_sim.shape[0]
    
    # Handle edge case
    if C < 2:
        print("Warning: Need at least 2 classes for meaningful inter-class analysis")
        return
    
    class_labels = [str(i) for i in range(C)]

    # Mask diagonal (self-similarity) for summary stats
    mask = ~np.eye(C, dtype=bool)
    off_diag = inter_class_sim[mask]

    # Per-class summaries
    # mean similarity to *other* classes
    row_sums = inter_class_sim.sum(axis=1) - np.diag(inter_class_sim)  # subtract self
    mean_to_others = row_sums / (C - 1)
    
    # nearest neighbor similarity (max excluding self)
    max_to_other = inter_class_sim.copy()
    np.fill_diagonal(max_to_other, -np.inf)
    nn_sim = max_to_other.max(axis=1)
    nn_idx = max_to_other.argmax(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1) Heatmap with diagonal blanked
    # Create masked array for better handling of NaN values
    heatmap_data = np.ma.masked_where(np.eye(C, dtype=bool), inter_class_sim)
    im = axes[0, 0].imshow(heatmap_data, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Inter-class Similarity Heatmap (diagonal masked)', fontsize=12)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Class')
    
    if C <= 30:  # only label if not too crowded
        axes[0, 0].set_xticks(range(C))
        axes[0, 0].set_yticks(range(C))
        axes[0, 0].set_xticklabels(class_labels, rotation=90)
        axes[0, 0].set_yticklabels(class_labels)
    
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar.set_label('Similarity', rotation=270, labelpad=14)

    # 2) Histogram of off-diagonal similarities
    axes[0, 1].hist(off_diag, bins=min(40, len(off_diag)//2), edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of Inter-class Similarities (off-diagonal)', fontsize=12)
    axes[0, 1].set_xlabel('Similarity')
    axes[0, 1].set_ylabel('Count')
    
    mean_val = off_diag.mean()
    axes[0, 1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    axes[0, 1].legend()

    # 3) Per-class average similarity to others (who is most generic?)
    order_avg = np.argsort(mean_to_others)[::-1]  # high to low
    bars1 = axes[1, 0].bar(range(C), mean_to_others[order_avg], alpha=0.7)
    axes[1, 0].set_title('Per-class Mean Similarity to Other Classes', fontsize=12)
    axes[1, 0].set_xlabel('Class (sorted by avg similarity)')
    axes[1, 0].set_ylabel('Mean similarity to others')
    
    if C <= 30:
        axes[1, 0].set_xticks(range(C))
        axes[1, 0].set_xticklabels([class_labels[i] for i in order_avg], rotation=90)
    
    # Add value labels on bars if not too many
    if C <= 20:
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 4) Per-class nearest neighbor similarity (who is most confusable?)
    order_nn = np.argsort(nn_sim)[::-1]
    bars2 = axes[1, 1].bar(range(C), nn_sim[order_nn], alpha=0.7, color='orange')
    axes[1, 1].set_title('Per-class Nearest-Neighbor Similarity', fontsize=12)
    axes[1, 1].set_xlabel('Class (sorted by max similarity)')
    axes[1, 1].set_ylabel('Max similarity to another class')
    
    if C <= 30:
        # Create labels showing class -> nearest neighbor
        xt_labels = []
        for i in order_nn:
            nn_class = nn_idx[i]
            xt_labels.append(f"{class_labels[i]}→{class_labels[nn_class]}")
        
        axes[1, 1].set_xticks(range(C))
        axes[1, 1].set_xticklabels(xt_labels, rotation=90)
    
    # Add value labels on bars if not too many
    if C <= 20:
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()




def plot_intra_class_embedding(
    df: pd.DataFrame,
    descriptors: Dict[int, np.ndarray],
    plot_path: str,
    classes: Optional[List[int]] = None,
    means: Optional[List[float]] = None,
    stds: Optional[List[float]] = None
):
    """
    Plot PCA projections of class embeddings with enhanced visual styling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'class_id' column.
    descriptors : Dict[int, np.ndarray] or np.ndarray
        Dictionary mapping DataFrame indices to descriptor vectors (or a 2D array with row indices matching df).
    plot_path : str
        Path to save the resulting plot.
    classes : Optional[List[int]]
        List of class_ids to plot. If None, sample up to 5 random classes.
    means : Optional[List[float]]
        Optional list of mean similarities per class (for annotation).
    stds : Optional[List[float]]
        Optional list of std similarities per class (for annotation).
    """
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    # Set modern style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Enhanced color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#F4A460', '#87CEEB', '#FFB6C1', '#98FB98']
    
    unique_class_ids = np.unique(df['class_id'])
    if classes is None:
        classes = np.random.choice(unique_class_ids, size=min(5, len(unique_class_ids)), replace=False)
    else:
        classes = np.array(classes)

    n_classes = len(classes)
    n_cols = min(3, n_classes)
    n_rows = math.ceil(n_classes / n_cols)

    # Create figure with better proportions and spacing
    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows), facecolor='white')
    fig.suptitle('Class Embedding PCA Projections', fontsize=16, fontweight='bold', 
                 y=0.95, color='#2C3E50')

    # Collect all PCA results first to determine global axis limits
    all_proj = []
    per_class_proj = {}
    for idx, class_id in enumerate(classes):
        class_idxs = df[df['class_id'] == class_id].index
        if isinstance(descriptors, dict):
            class_descriptors = np.stack([descriptors[i] for i in class_idxs])
        else:
            class_descriptors = descriptors[class_idxs]

        if class_descriptors.shape[0] >= 2:
            pca = PCA(n_components=2)
            Y = pca.fit_transform(class_descriptors)
            per_class_proj[class_id] = Y
            all_proj.append(Y)

    if all_proj:
        all_proj = np.vstack(all_proj)
        x_min, x_max = all_proj[:, 0].min(), all_proj[:, 0].max()
        y_min, y_max = all_proj[:, 1].min(), all_proj[:, 1].max()
        # Add more generous padding
        pad_x = 0.1 * (x_max - x_min) if x_max != x_min else 1
        pad_y = 0.1 * (y_max - y_min) if y_max != y_min else 1
        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_y, y_max + pad_y)
    else:
        xlim, ylim = (-1, 1), (-1, 1)

    # Create subplots with better spacing
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25)

    # Now plot each class with enhanced styling
    for idx, class_id in enumerate(classes):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        
        # Set background color
        ax.set_facecolor('#FAFAFA')
        
        if class_id not in per_class_proj:
            # Styled message for insufficient data
            ax.text(0.5, 0.5, "Insufficient Data\n(< 2 samples)", 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE5E5', 
                            edgecolor='#FF9999', alpha=0.8),
                   color='#CC0000')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            Y = per_class_proj[class_id]
            color = colors[idx % len(colors)]
            
            # Create scatter plot with enhanced styling
            scatter = ax.scatter(Y[:, 0], Y[:, 1], 
                               c=color, alpha=0.7, s=60, 
                               edgecolors='white', linewidths=1.5,
                               zorder=5)
            
            # Add subtle density contours if enough points
            if len(Y) > 10:
                try:
                    x_contour = np.linspace(xlim[0], xlim[1], 50)
                    y_contour = np.linspace(ylim[0], ylim[1], 50)
                    X_contour, Y_contour = np.meshgrid(x_contour, y_contour)
                    
                    # Simple 2D histogram for contours
                    from scipy.stats import gaussian_kde
                    if len(Y) > 1:
                        kde = gaussian_kde(Y.T)
                        density = kde(np.vstack([X_contour.ravel(), Y_contour.ravel()]))
                        density = density.reshape(X_contour.shape)
                        ax.contour(X_contour, Y_contour, density, levels=3, 
                                 colors=color, alpha=0.3, linewidths=1)
                except:
                    pass  # Skip contours if any error occurs
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Enhanced title with statistics
        if means is not None and stds is not None and class_id in per_class_proj:
            try:
                mean_val = means[class_id] if isinstance(means, dict) else means[idx] if len(means) == n_classes else means[class_id]
                std_val = stds[class_id] if isinstance(stds, dict) else stds[idx] if len(stds) == n_classes else stds[class_id]
                title_text = f'Class {class_id}\n mean = {mean_val:.3f} • std = {std_val:.3f}'
            except:
                title_text = f'Class {class_id}'
        else:
            title_text = f'Class {class_id}'
            
        ax.set_title(title_text, fontsize=11, fontweight='bold', 
                    color='#2C3E50', pad=15)

        # Enhanced axis labels
        ax.set_xlabel('PC1', fontsize=10, color='#34495E', fontweight='500')
        ax.set_ylabel('PC2', fontsize=10, color='#34495E', fontweight='500')
        
        # Modern grid styling
        ax.grid(True, linestyle='-', alpha=0.9, color='#BDC3C7', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Enhanced spines
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.2)
        
        # Style tick labels
        ax.tick_params(colors='#7F8C8D', labelsize=9)
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Add subtle border effect
        border = patches.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                                 linewidth=2, edgecolor='#E8E8E8', facecolor='none', zorder=1)
        ax.add_patch(border)

    # Hide unused subplots with style
    for idx in range(n_classes, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        ax.set_facecolor('white')

    # Add a subtle footer
    fig.text(0.99, 0.01, f'Generated with {n_classes} classes • PCA Dimensionality Reduction', 
             ha='right', va='bottom', fontsize=8, color='#95A5A6', style='italic')

    # Save with high quality
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                metadata={'Title': 'Class Embedding PCA Projections'})
    plt.close()
    


def plot_inter_class_embedding(dataconfig: pd.DataFrame, descriptors: np.ndarray, 
                                       plot_path: str, show_individual_points=True):
    """
    Advanced version that shows both individual samples and class centroids with 2D Gaussian distributions.
    """
    from matplotlib.patches import Ellipse
    
    class_ids = sorted(dataconfig['class_id'].unique())
    n_classes = len(class_ids)
    
    # Auto-prune if too many classes for meaningful visualization
    if n_classes > 100:
        # First, filter out very small classes (< 5 samples)
        class_counts = dataconfig['class_id'].value_counts()
        large_classes = class_counts[class_counts >= 5].index
        
        # If still too many, randomly sample 500 classes
        if len(large_classes) > 100:
            np.random.seed(42)  # For reproducibility
            selected_classes = np.random.choice(large_classes, size=100, replace=False)
        else:
            selected_classes = large_classes
        
        # Filter data to selected classes
        mask = dataconfig['class_id'].isin(selected_classes)
        dataconfig = dataconfig[mask]
        descriptors = descriptors[mask]
        class_ids = sorted(dataconfig['class_id'].unique())
        n_classes = len(class_ids)
        print(f"Auto-pruned to {n_classes} classes for better visualization")
    
    # Reduce dimensionality for all points
    pca = PCA(n_components=2, random_state=42)
    Y_all = pca.fit_transform(descriptors)
    
    # Compute centroids in reduced space
    centroids = []
    for class_id in class_ids:
        class_mask = dataconfig['class_id'] == class_id
        centroid = Y_all[class_mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # Create plot with adaptive sizing
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Smart color and styling for many classes
    if n_classes > 50:
        colors = plt.cm.hsv(np.linspace(0, 1, n_classes))
        point_size = max(5, 20 - n_classes//50)  # Smaller points for more classes
        point_alpha = max(0.1, 0.4 - n_classes/5000)  # More transparent for more classes
        ellipse_alpha = max(0.05, 0.2 - n_classes/10000)
        show_legend = False
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_classes] if n_classes <= 10 else plt.cm.hsv(np.linspace(0, 1, n_classes))
        point_size = 20
        point_alpha = 0.3
        ellipse_alpha = 0.3
        show_legend = n_classes <= 20  # Only show legend for reasonable number of classes
    
    # Plot 2D Gaussians for each class
    for i, class_id in enumerate(class_ids):
        class_mask = dataconfig['class_id'] == class_id
        class_points = Y_all[class_mask]
        
        if len(class_points) > 1:  # Need at least 2 points for covariance
            cov = np.cov(class_points.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Calculate rotation angle
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            # Draw single confidence ellipse (1-sigma)
            width = 2 * np.sqrt(eigenvals[0])
            height = 2 * np.sqrt(eigenvals[1])
            
            ellipse = Ellipse(centroids[i], width, height, angle=angle,
                            alpha=ellipse_alpha, facecolor=colors[i], edgecolor=colors[i],
                            linewidth=1.5)
            ax.add_patch(ellipse)
    
    # Plot individual points (with adaptive styling)
    if show_individual_points:
        for i, class_id in enumerate(class_ids):
            class_mask = dataconfig['class_id'] == class_id
            class_points = Y_all[class_mask]
            label = f'Class {class_id}' if show_legend else None
            ax.scatter(class_points[:, 0], class_points[:, 1], 
                      c=[colors[i]], alpha=point_alpha, s=point_size, 
                      label=label)
    

    
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12, fontweight='bold')
    
    title = f'Inter-Class Embeddings: {n_classes} Classes\nTotal Variance Explained: {var_explained.sum():.1%}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Show legend only for reasonable number of classes
    if show_legend and show_individual_points:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()