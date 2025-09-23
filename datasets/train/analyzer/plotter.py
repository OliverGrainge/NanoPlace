import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_intra_utmdist(stats: dict, plotting_dir: str):
    """
    Plot histogram distributions of intra-class distance statistics.

    Args:
        stats: Dictionary containing 'class_mean_dist' and 'class_std_dist'
        plotting_dir: Directory to save the plot
    """
    means = stats.get("intra_utmdist_mean", [])
    stds = stats.get("intra_utmdist_std", [])

    if len(means) == 0:
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot histogram of mean distances
    ax1.hist(
        means,
        bins=min(20, len(means) // 2 + 1),
        alpha=0.7,
        color="skyblue",
        edgecolor="navy",
        linewidth=1.2,
    )
    ax1.set_xlabel("Mean Distance (UTM units)", fontsize=12)
    ax1.set_ylabel("Number of Classes", fontsize=12)
    ax1.set_title(
        "Distribution of Intra-Class\nMean Distances", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Add summary statistics for means
    mean_of_means = np.mean(means)
    std_of_means = np.std(means)
    ax1.axvline(
        mean_of_means,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean: {mean_of_means:.2f}",
    )
    ax1.axvline(
        mean_of_means + std_of_means,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"±1 SD: {std_of_means:.2f}",
    )
    ax1.axvline(mean_of_means - std_of_means, color="red", linestyle=":", alpha=0.7)
    ax1.legend(fontsize=10)

    # Plot histogram of standard deviations
    ax2.hist(
        stds,
        bins=min(20, len(stds) // 2 + 1),
        alpha=0.7,
        color="lightcoral",
        edgecolor="darkred",
        linewidth=1.2,
    )
    ax2.set_xlabel("Standard Deviation (UTM units)", fontsize=12)
    ax2.set_ylabel("Number of Classes", fontsize=12)
    ax2.set_title(
        "Distribution of Intra-Class\nDistance Standard Deviations",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    # Add summary statistics for stds
    mean_of_stds = np.mean(stds)
    std_of_stds = np.std(stds)
    ax2.axvline(
        mean_of_stds,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean: {mean_of_stds:.2f}",
    )
    ax2.axvline(
        mean_of_stds + std_of_stds,
        color="darkred",
        linestyle=":",
        alpha=0.7,
        label=f"±1 SD: {std_of_stds:.2f}",
    )
    ax2.axvline(mean_of_stds - std_of_stds, color="darkred", linestyle=":", alpha=0.7)
    ax2.legend(fontsize=10)

    # Add overall title and adjust layout
    fig.suptitle(
        f"Intra-Class Distance Statistics (n = {len(means)} classes)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(plotting_dir, "intra_utmdist.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_inter_utmdist(stats: dict, plotting_dir: str):
    """
    Plot histogram distributions of inter-class distance statistics.

    Args:
        stats: Dictionary containing 'inter_utmdist_matrix' - square matrix with distances between classes
        plotting_dir: Directory to save the plot
    """
    inter_matrix = stats.get("inter_utmdist_matrix", None)

    if inter_matrix is None or len(inter_matrix) == 0:
        return

    # Convert to numpy array if not already
    inter_matrix = np.array(inter_matrix)

    # Extract upper triangular part (excluding diagonal) to get all pairwise distances
    triu_indices = np.triu_indices_from(inter_matrix, k=1)
    inter_distances = inter_matrix[triu_indices]

    if len(inter_distances) == 0:
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot histogram of inter-class distances
    ax.hist(
        inter_distances,
        bins=min(30, len(inter_distances) // 3 + 1),
        alpha=0.7,
        color="lightgreen",
        edgecolor="darkgreen",
        linewidth=1.2,
    )
    ax.set_xlabel("Distance Between Class Centroids (UTM units)", fontsize=12)
    ax.set_ylabel("Number of Class Pairs", fontsize=12)
    ax.set_title(
        "Distribution of Inter-Class\nCentroid Distances",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add summary statistics
    mean_inter_dist = np.mean(inter_distances)
    std_inter_dist = np.std(inter_distances)
    median_inter_dist = np.median(inter_distances)

    ax.axvline(
        mean_inter_dist,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_inter_dist:.2f}",
    )
    ax.axvline(
        median_inter_dist,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_inter_dist:.2f}",
    )
    ax.axvline(
        mean_inter_dist + std_inter_dist,
        color="darkgreen",
        linestyle=":",
        alpha=0.7,
        label=f"±1 SD: {std_inter_dist:.2f}",
    )
    ax.axvline(
        mean_inter_dist - std_inter_dist, color="darkgreen", linestyle=":", alpha=0.7
    )

    ax.legend(fontsize=10)

    # Add overall title
    fig.suptitle(
        f"Inter-Class Distance Statistics", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(plotting_dir, "inter_utmdist.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_class_samples(stats: dict, plotting_dir: str):
    """
    Plot samples from each class in rows, with up to 6 classes and 6 images per class.
    Images in the same row (class) have no whitespace between them.

    Args:
        stats: Dictionary containing class_samples with structure:
               stats["class_samples"][class_id] = list_of_image_paths
        plotting_dir: Directory to save the plot
    """
    if "class_samples" not in stats:
        return
    class_samples = stats["class_samples"]

    # Limit to maximum 6 classes
    class_ids = list(class_samples.keys())[:6]

    if not class_ids:
        print("No class samples found to plot.")
        return

    max_images_per_class = min(
        6, max(len(class_samples[class_id]) for class_id in class_ids)
    )
    fig, axes = plt.subplots(
        len(class_ids),
        max_images_per_class,
        figsize=(max_images_per_class * 2, len(class_ids) * 2),
    )

    if len(class_ids) == 1:
        axes = axes.reshape(1, -1)
    elif max_images_per_class == 1:
        axes = axes.reshape(-1, 1)

    plt.subplots_adjust(wspace=0, hspace=0.1)

    for row_idx, class_id in enumerate(class_ids):
        image_paths = class_samples[class_id]

        for col_idx in range(max_images_per_class):
            ax = axes[row_idx, col_idx] if len(class_ids) > 1 else axes[col_idx]

            if col_idx < len(image_paths):
                # Load and display image
                try:
                    img = Image.open(image_paths[col_idx])
                    ax.imshow(img)

                    # Add class label only to the first image of each row
                    if col_idx == 0:
                        ax.set_ylabel(
                            f"Class {class_id}",
                            rotation=0,
                            labelpad=50,
                            verticalalignment="center",
                        )

                except Exception as e:
                    print(f"Error loading image {image_paths[col_idx]}: {e}")
                    ax.text(0.5, 0.5, "Image\nError", ha="center", va="center")
            else:
                # Hide empty subplots
                ax.set_visible(False)

            # Remove axes ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove the frame around each image
            for spine in ax.spines.values():
                spine.set_visible(False)
    plt.suptitle("Class Samples", fontsize=16, y=0.98)
    output_path = os.path.join(plotting_dir, "class_samples.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150, pad_inches=0.1)


def plot_class_counts(stats: dict, plotting_dir: str):
    """
    Shows class distribution with clear summary statistics
    """
    class_counts = stats["class_counts"]
    counts = list(class_counts.values())
    class_ids = list(class_counts.keys())
    
    # Calculate summary statistics
    num_classes = len(class_counts)
    total_samples = sum(counts)
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    std_count = np.std(counts)
    min_count = min(counts)
    max_count = max(counts)
    
    # Create figure with single histogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Histogram of counts
    n_bins = min(30, max(10, num_classes // 5))
    n, bins, patches = ax.hist(
        counts,
        bins=n_bins,
        alpha=0.7,
        color="skyblue",
        edgecolor="navy",
        linewidth=1
    )
    
    # Add statistical lines
    ax.axvline(
        x=mean_count,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_count:.1f}"
    )
    ax.axvline(
        x=median_count,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_count:.1f}"
    )
    
    # Add labels and title
    ax.set_xlabel("Number of Samples per Class", fontsize=12)
    ax.set_ylabel("Number of Classes", fontsize=12)
    ax.set_title("Class Distribution Overview", fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics as text box
    stats_text = (
        f"Dataset Summary:\n"
        f"• Total Classes: {num_classes:,}\n"
        f"• Total Samples: {total_samples:,}\n"
        f"• Avg Samples/Class: {mean_count:.1f}\n"
        f"• Std Deviation: {std_count:.1f}\n"
        f"• Min Samples: {min_count}\n"
        f"• Max Samples: {max_count}\n"
        f"• Balance Ratio: {min_count/max_count:.3f}"
    )
    
    # Position text box in upper right
    ax.text(
        0.98, 0.98, 
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10,
        fontfamily='monospace'
    )
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(plotting_dir, "class_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    
    # Print summary to console as well
    print(f"\n{'='*50}")
    print(f"CLASS DISTRIBUTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total Classes: {num_classes:,}")
    print(f"Total Samples: {total_samples:,}")
    print(f"Average Samples per Class: {mean_count:.2f}")
    print(f"Standard Deviation: {std_count:.2f}")
    print(f"Range: {min_count} - {max_count} samples")
    print(f"Balance Ratio (min/max): {min_count/max_count:.3f}")
    print(f"{'='*50}\n")


class Plotter:
    def __init__(self, plotting_dir: str):
        self.plotting_dir = plotting_dir
        os.makedirs(self.plotting_dir, exist_ok=True)
        self.stats = {}

    def plot(self):
        plot_intra_utmdist(self.stats, self.plotting_dir)
        plot_inter_utmdist(self.stats, self.plotting_dir)
        plot_class_samples(self.stats, self.plotting_dir)
        plot_class_counts(self.stats, self.plotting_dir)

    def add_stats(self, stats: dict):
        self.stats.update(stats)
