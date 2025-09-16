"""
Utilities for analyzing spatial dataset configurations.

Usage:
    # Simple usage with parquet file
    results = analyze_config_file("configs/nanoplace_basic/config.parquet")
    
    # Or with DataFrame and directory
    df = pd.read_parquet("configs/nanoplace_basic/config.parquet")
    results = analyze_spatial_dataset(df, "configs/nanoplace_basic", "configs/nanoplace_basic/metadata.json")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json


def analyze_spatial_dataset(df: pd.DataFrame, 
                          config_dir: str,
                          config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a spatial dataset configuration DataFrame and save results to config directory.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: image_path, easting, northing, class_ids
    config_dir : str
        Directory where the config files are saved (where analysis results will be saved)
    config_path : str, optional
        Path to the config metadata JSON file for additional context
        
    Returns
    -------
    analysis_results : dict
        Dictionary containing analysis results and statistics
    """
    
    # Load metadata if available
    metadata = {}
    if config_path:
        try:
            with open(config_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"Metadata file not found: {config_path}")
    
    # Basic dataset statistics
    n_images = len(df)
    n_classes = df['class_ids'].nunique()
    
    # Spatial extent
    easting_range = (df['easting'].min(), df['easting'].max())
    northing_range = (df['northing'].min(), df['northing'].max())
    spatial_extent = {
        'easting_range': easting_range,
        'northing_range': northing_range,
        'width_m': easting_range[1] - easting_range[0],
        'height_m': northing_range[1] - northing_range[0]
    }
    
    # Class statistics
    class_counts = df['class_ids'].value_counts().sort_index()
    class_stats = {
        'mean_images_per_class': class_counts.mean(),
        'median_images_per_class': class_counts.median(),
        'min_images_per_class': class_counts.min(),
        'max_images_per_class': class_counts.max(),
        'std_images_per_class': class_counts.std()
    }
    
    # Print summary
    print("=" * 60)
    print("SPATIAL DATASET ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {metadata.get('dataset_folder', 'Unknown')}")
    print(f"Total images: {n_images:,}")
    print(f"Total classes: {n_classes:,}")
    print(f"Max class distance: {metadata.get('class_max_distance', 'Unknown')} m")
    print()
    
    print("SPATIAL EXTENT:")
    print(f"  Easting range: {easting_range[0]:.1f} - {easting_range[1]:.1f} m")
    print(f"  Northing range: {northing_range[0]:.1f} - {northing_range[1]:.1f} m")
    print(f"  Total width: {spatial_extent['width_m']:.1f} m")
    print(f"  Total height: {spatial_extent['height_m']:.1f} m")
    print()
    
    print("CLASS STATISTICS:")
    print(f"  Mean images per class: {class_stats['mean_images_per_class']:.1f}")
    print(f"  Median images per class: {class_stats['median_images_per_class']:.1f}")
    print(f"  Min images per class: {class_stats['min_images_per_class']}")
    print(f"  Max images per class: {class_stats['max_images_per_class']}")
    print(f"  Std images per class: {class_stats['std_images_per_class']:.1f}")
    print()
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Spatial distribution with classes
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df['easting'], df['northing'], 
                         c=df['class_ids'], cmap='tab20', 
                         alpha=0.6, s=10)
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_title(f'Spatial Distribution by Class\n({n_classes} classes, {n_images} images)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Class size distribution
    ax2 = plt.subplot(2, 3, 2)
    bins = np.arange(0.5, class_counts.max() + 1.5, 1)
    ax2.hist(class_counts.values, bins=bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Images per Class')
    ax2.set_ylabel('Number of Classes')
    ax2.set_title('Distribution of Class Sizes')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative class size distribution
    ax3 = plt.subplot(2, 3, 3)
    sorted_counts = np.sort(class_counts.values)[::-1]
    cumulative_images = np.cumsum(sorted_counts)
    ax3.plot(range(1, len(sorted_counts) + 1), cumulative_images / n_images * 100)
    ax3.set_xlabel('Class Rank')
    ax3.set_ylabel('Cumulative % of Images')
    ax3.set_title('Cumulative Image Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Spatial density heatmap
    ax4 = plt.subplot(2, 3, 4)
    h = ax4.hist2d(df['easting'], df['northing'], bins=50, cmap='Blues')
    ax4.set_xlabel('Easting (m)')
    ax4.set_ylabel('Northing (m)')
    ax4.set_title('Spatial Density Heatmap')
    plt.colorbar(h[3], ax=ax4, label='Image Count')
    ax4.set_aspect('equal')
    
    # 5. Class centroids
    ax5 = plt.subplot(2, 3, 5)
    class_centroids = df.groupby('class_ids')[['easting', 'northing']].mean()
    ax5.scatter(class_centroids['easting'], class_centroids['northing'], 
               c=class_centroids.index, cmap='tab20', s=30, alpha=0.8)
    ax5.set_xlabel('Easting (m)')
    ax5.set_ylabel('Northing (m)')
    ax5.set_title(f'Class Centroids\n({len(class_centroids)} classes)')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Distance statistics within classes
    ax6 = plt.subplot(2, 3, 6)
    max_distances = []
    for class_id in df['class_ids'].unique():
        class_data = df[df['class_ids'] == class_id]
        if len(class_data) > 1:
            coords = class_data[['easting', 'northing']].values
            distances = []
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                    distances.append(dist)
            max_distances.append(max(distances) if distances else 0)
    
    if max_distances:
        ax6.hist(max_distances, bins=20, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Maximum Distance within Class (m)')
        ax6.set_ylabel('Number of Classes')
        ax6.set_title('Within-Class Maximum Distances')
        ax6.grid(True, alpha=0.3)
        ax6.axvline(metadata.get('class_max_distance', 15), 
                   color='red', linestyle='--', 
                   label=f"Max distance threshold: {metadata.get('class_max_distance', 15)}m")
        ax6.legend()
    
    plt.tight_layout()
    
    # Save plots to config directory
    config_path_obj = Path(config_dir)
    plot_path = config_path_obj / 'spatial_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Compile results
    analysis_results = {
        'basic_stats': {
            'n_images': n_images,
            'n_classes': n_classes,
            'dataset_folder': metadata.get('dataset_folder'),
            'class_max_distance': metadata.get('class_max_distance')
        },
        'spatial_extent': spatial_extent,
        'class_stats': class_stats,
        'class_counts': class_counts.to_dict(),
        'within_class_max_distances': max_distances,
        'metadata': metadata
    }
    
    # Save analysis results to config directory
    analysis_path = config_path_obj / 'analysis_results.json'
    with open(analysis_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(analysis_results, f, indent=2, default=convert_numpy)
    
    print(f"Analysis plot saved to: {plot_path}")
    print(f"Analysis results saved to: {analysis_path}")
    
    return analysis_results


def compare_spatial_configs(config_paths: list, labels: Optional[list] = None):
    """
    Compare multiple spatial dataset configurations.
    
    Parameters
    ----------
    config_paths : list of str
        Paths to parquet files containing spatial configurations
    labels : list of str, optional
        Labels for each configuration (defaults to config filenames)
    """
    
    if labels is None:
        labels = [Path(p).stem for p in config_paths]
    
    configs = []
    for path in config_paths:
        df = pd.read_parquet(path)
        configs.append(df)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Images per dataset
    ax1 = axes[0, 0]
    n_images = [len(df) for df in configs]
    ax1.bar(labels, n_images)
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Total Images per Dataset')
    ax1.tick_params(axis='x', rotation=45)
    
    # Classes per dataset
    ax2 = axes[0, 1]
    n_classes = [df['class_ids'].nunique() for df in configs]
    ax2.bar(labels, n_classes)
    ax2.set_ylabel('Number of Classes')
    ax2.set_title('Total Classes per Dataset')
    ax2.tick_params(axis='x', rotation=45)
    
    # Average images per class
    ax3 = axes[1, 0]
    avg_per_class = [len(df) / df['class_ids'].nunique() for df in configs]
    ax3.bar(labels, avg_per_class)
    ax3.set_ylabel('Average Images per Class')
    ax3.set_title('Average Images per Class')
    ax3.tick_params(axis='x', rotation=45)
    
    # Spatial extent comparison
    ax4 = axes[1, 1]
    extents = []
    for df in configs:
        width = df['easting'].max() - df['easting'].min()
        height = df['northing'].max() - df['northing'].min()
        area = width * height / 1e6  # Convert to km²
        extents.append(area)
    
    ax4.bar(labels, extents)
    ax4.set_ylabel('Spatial Extent (km²)')
    ax4.set_title('Spatial Coverage')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.close()  # Close figure to free memory
    
    # Print comparison table
    print("\nCOMPARISON SUMMARY:")
    print("-" * 80)
    print(f"{'Dataset':<20} {'Images':<10} {'Classes':<10} {'Avg/Class':<12} {'Extent (km²)':<15}")
    print("-" * 80)
    
    for i, (label, df) in enumerate(zip(labels, configs)):
        n_img = len(df)
        n_cls = df['class_ids'].nunique()
        avg_cls = n_img / n_cls
        extent = extents[i]
        print(f"{label:<20} {n_img:<10} {n_cls:<10} {avg_cls:<12.1f} {extent:<15.2f}")


def analyze_dataset(parquet_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze a spatial dataset configuration from a parquet file.
    Automatically finds the config directory and metadata file.
    
    Parameters
    ----------
    parquet_path : str
        Path to the parquet configuration file
        
    Returns
    -------
    analysis_results : dict
        Dictionary containing analysis results and statistics
    """
    parquet_path_obj = Path(parquet_path)
    config_dir = parquet_path_obj.parent
    
    # Look for metadata file in the same directory
    metadata_path = config_dir / 'metadata.json'
    metadata_path_str = str(metadata_path) if metadata_path.exists() else None
    
    # Load the dataframe
    df = pd.read_parquet(parquet_path)
    
    # Run analysis
    return analyze_spatial_dataset(df, str(config_dir), metadata_path_str)


def export_class_summary(df: pd.DataFrame, output_path: str):
    """
    Export a detailed summary of each class to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        Spatial dataset DataFrame
    output_path : str
        Path to save the CSV summary
    """
    
    class_summary = []
    
    for class_id in sorted(df['class_ids'].unique()):
        class_data = df[df['class_ids'] == class_id]
        
        # Basic stats
        n_images = len(class_data)
        centroid_easting = class_data['easting'].mean()
        centroid_northing = class_data['northing'].mean()
        
        # Spatial extent within class
        easting_range = class_data['easting'].max() - class_data['easting'].min()
        northing_range = class_data['northing'].max() - class_data['northing'].min()
        
        # Maximum distance within class
        if n_images > 1:
            coords = class_data[['easting', 'northing']].values
            max_distance = 0
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                    max_distance = max(max_distance, dist)
        else:
            max_distance = 0
        
        class_summary.append({
            'class_id': class_id,
            'n_images': n_images,
            'centroid_easting': centroid_easting,
            'centroid_northing': centroid_northing,
            'easting_range': easting_range,
            'northing_range': northing_range,
            'max_distance': max_distance,
            'spatial_area': easting_range * northing_range
        })
    
    summary_df = pd.DataFrame(class_summary)
    summary_df.to_csv(output_path, index=False)
    print(f"Class summary exported to: {output_path}")
    
    return summary_df