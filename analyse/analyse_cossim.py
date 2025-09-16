from eval.utils import correct_matches, match_cosine


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def analyse_cossim(query_descriptors: np.ndarray, database_descriptors: np.ndarray, ground_truth: np.ndarray, save_path: str = None):
    """
    Analyze and plot the distribution of cosine similarities for correct vs incorrect matches.
    """
    print(f"Analyzing cosine similarity distribution for {len(query_descriptors)} query descriptors and {len(database_descriptors)} database descriptors")
    # Get matches and scores
    matches, scores = match_cosine(query_descriptors=query_descriptors, database_descriptors=database_descriptors, k=1)
    scores = scores.flatten()
    correct = correct_matches(matches, ground_truth, k=1)
    
    correct_scores = scores[correct]
    incorrect_scores = scores[~correct]
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cosine Similarity Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overlaid histograms
    ax1 = axes[0, 0]
    bins = np.linspace(min(scores), max(scores), 50)
    
    ax1.hist(incorrect_scores, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect_scores)})', 
             color='lightcoral', density=True)
    ax1.hist(correct_scores, bins=bins, alpha=0.7, label=f'Correct (n={len(correct_scores)})', 
             color='lightblue', density=True)
    
    ax1.axvline(np.mean(correct_scores), color='blue', linestyle='--', alpha=0.8, 
                label=f'Correct mean: {np.mean(correct_scores):.3f}')
    ax1.axvline(np.mean(incorrect_scores), color='red', linestyle='--', alpha=0.8,
                label=f'Incorrect mean: {np.mean(incorrect_scores):.3f}')
    
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    box_data = [incorrect_scores, correct_scores]
    bp = ax2.boxplot(box_data, labels=['Incorrect', 'Correct'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    
    # Sort scores for CDF
    correct_sorted = np.sort(correct_scores)
    incorrect_sorted = np.sort(incorrect_scores)
    
    # Calculate cumulative probabilities
    correct_cdf = np.arange(1, len(correct_sorted) + 1) / len(correct_sorted)
    incorrect_cdf = np.arange(1, len(incorrect_sorted) + 1) / len(incorrect_sorted)
    
    ax3.plot(correct_sorted, correct_cdf, label='Correct', color='blue', linewidth=2)
    ax3.plot(incorrect_sorted, incorrect_cdf, label='Incorrect', color='red', linewidth=2)
    
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    correct_stats = {
        'Mean': np.mean(correct_scores),
        'Median': np.median(correct_scores),
        'Std': np.std(correct_scores),
        'Min': np.min(correct_scores),
        'Max': np.max(correct_scores),
        'Q25': np.percentile(correct_scores, 25),
        'Q75': np.percentile(correct_scores, 75)
    }
    
    incorrect_stats = {
        'Mean': np.mean(incorrect_scores),
        'Median': np.median(incorrect_scores),
        'Std': np.std(incorrect_scores),
        'Min': np.min(incorrect_scores),
        'Max': np.max(incorrect_scores),
        'Q25': np.percentile(incorrect_scores, 25),
        'Q75': np.percentile(incorrect_scores, 75)
    }
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(correct_scores, incorrect_scores)
    ks_stat, ks_p_value = stats.ks_2samp(correct_scores, incorrect_scores)
    
    # Create summary text
    summary_text = f"""
    SUMMARY STATISTICS
    
    Correct Matches (n={len(correct_scores)}):
    Mean: {correct_stats['Mean']:.4f}
    Median: {correct_stats['Median']:.4f}
    Std: {correct_stats['Std']:.4f}
    Range: [{correct_stats['Min']:.4f}, {correct_stats['Max']:.4f}]
    IQR: [{correct_stats['Q25']:.4f}, {correct_stats['Q75']:.4f}]
    
    Incorrect Matches (n={len(incorrect_scores)}):
    Mean: {incorrect_stats['Mean']:.4f}
    Median: {incorrect_stats['Median']:.4f}
    Std: {incorrect_stats['Std']:.4f}
    Range: [{incorrect_stats['Min']:.4f}, {incorrect_stats['Max']:.4f}]
    IQR: [{incorrect_stats['Q25']:.4f}, {incorrect_stats['Q75']:.4f}]
    
    STATISTICAL TESTS
    T-test: t={t_stat:.4f}, p={p_value:.4e}
    KS-test: D={ks_stat:.4f}, p={ks_p_value:.4e}
    
    Accuracy: {len(correct_scores)/(len(correct_scores)+len(incorrect_scores))*100:.2f}%
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to /tmp directory instead of showing it
    import os
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()  # Close the figure to free memory
    
    # Print summary statistics to console
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Correct matches: {len(correct_scores)}")
    print(f"Incorrect matches: {len(incorrect_scores)}")
    print(f"Accuracy: {len(correct_scores)/(len(correct_scores)+len(incorrect_scores))*100:.2f}%")
    print(f"Correct matches mean similarity: {np.mean(correct_scores):.4f}")
    print(f"Incorrect matches mean similarity: {np.mean(incorrect_scores):.4f}")
    print(f"Similarity difference: {np.mean(correct_scores) - np.mean(incorrect_scores):.4f}")
    print("=" * 25)
    
    return correct_scores, incorrect_scores
