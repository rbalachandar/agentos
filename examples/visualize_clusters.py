#!/usr/bin/env python3
"""
Reproduce Figure 3.2 from AgentOS paper: Attention Matrix Heatmap.

This version clearly shows semantic clusters (blocks) along the diagonal,
demonstrating how tokens aggregate into coherent semantic slices.

Key improvements:
- Shows last layer attention (not aggregated) to preserve block structures
- Adds colored rectangles highlighting semantic clusters
- Uses clustering-style visualization
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from agentos import create_kernel


def find_attention_blocks(
    attention: np.ndarray,
    min_block_size: int = 3,
) -> List[Tuple[int, int, int, int]]:
    """Find rectangular blocks in attention matrix using a simple heuristic.

    A block is where attention is consistently high for tokens within a slice.

    Args:
        attention: (seq_len, seq_len) attention matrix
        min_block_size: Minimum size of a block to detect

    Returns:
        List of (row_start, row_end, col_start, col_end) block rectangles
    """
    seq_len = attention.shape[0]
    blocks = []

    # Simple approach: find high-attention regions along diagonal
    for i in range(seq_len):
        for j in range(i, min(i + min_block_size, seq_len)):
            # Check if this region has high attention
            region = attention[i:j+1, i:j+1]
            if region.mean() > 0.15:  # Threshold for "high attention"
                blocks.append((i, j+1, i, j+1))

    return blocks


def plot_attention_with_clusters(
    attention_weights: np.ndarray,
    tokens: list[str],
    boundary_positions: list[int],
    densities: np.ndarray,
    title: str = "AgentOS Figure 3.2: Attention Matrix with Semantic Clusters",
) -> plt.Figure:
    """Plot attention matrix showing semantic clusters as blocks.

    This reproduces the style of Figure 3.2 from the paper, where you can clearly
    see rectangular blocks along the diagonal representing semantic slices.

    Args:
        attention_weights: (num_layers, num_heads, seq_len, seq_len) attention weights
        tokens: List of tokens
        boundary_positions: List of semantic boundary positions
        densities: (seq_len,) density values
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    seq_len = len(tokens)

    # Use LAST LAYER only (not aggregated) to preserve block structures
    # The paper shows that individual layers have clearer block patterns
    last_layer_attention = attention_weights[-1]  # (num_heads, seq_len, seq_len)

    # Aggregate across heads (mean) but keep layers separate initially
    attention_per_head = last_layer_attention  # (num_heads, seq_len, seq_len)
    aggregated = attention_per_head.mean(axis=0)  # (seq_len, seq_len)

    fig, (ax_heatmap, ax_density, ax_legend) = plt.subplots(
        3, 1,
        figsize=(16, 12),
        gridspec_kw={'height_ratios': [4, 1.5, 0.3]},
    )

    # Custom colormap for better block visibility
    cmap = plt.cm.viridis

    # Plot attention heatmap
    im = ax_heatmap.imshow(aggregated, cmap=cmap, aspect='equal', origin='upper',
                           vmin=0, vmax=np.percentile(aggregated, 95))

    # Add semantic slice boundaries
    for i, boundary in enumerate(boundary_positions):
        if boundary < seq_len:
            ax_heatmap.axhline(y=boundary, color='white', linewidth=2, linestyle='--', alpha=0.8)
            ax_heatmap.axvline(x=boundary, color='white', linewidth=2, linestyle='--', alpha=0.8)

    # Add colored rectangles to highlight semantic clusters/blocks
    # This makes the "cluster A", "cluster B" concept visual
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
               '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B500', '#FFB6C1']

    for i, boundary in enumerate(boundary_positions[:-1]):
        if i + 1 < len(boundary_positions):
            start = boundary
            end = boundary_positions[i + 1]
            if end <= seq_len:
                # Add semi-transparent rectangle to highlight the block
                rect = Rectangle((start, start), end - start, end - start,
                               linewidth=2, edgecolor=colors[i % len(colors)],
                               facecolor=colors[i % len(colors)], alpha=0.15)
                ax_heatmap.add_patch(rect)

                # Add cluster label
                if (end - start) >= 3:  # Only label if reasonable size
                    mid = (start + end) / 2
                    ax_heatmap.text(mid, mid, f"σ{i+1}",
                                  ha='center', va='center',
                                  fontsize=8, fontweight='bold',
                                  color=colors[i % len(colors)],
                                  bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              alpha=0.8,
                                              edgecolor=colors[i % len(colors)]))

    # Configure axes
    ax_heatmap.set_xlabel('Key Position', fontsize=11, fontweight='bold')
    ax_heatmap.set_ylabel('Query Position', fontsize=11, fontweight='bold')
    ax_heatmap.set_title(title, fontsize=13, fontweight='bold')

    # Show sparse ticks
    step = max(1, seq_len // 12)
    ax_heatmap.set_xticks(range(0, seq_len, step))
    ax_heatmap.set_yticks(range(0, seq_len, step))
    ax_heatmap.set_xticklabels([tokens[i][:10] if i < seq_len else str(i)
                                  for i in range(0, seq_len, step)],
                       rotation=45, ha='right', fontsize=7)
    ax_heatmap.set_yticklabels([tokens[i][:10] if i < seq_len else str(i)
                                  for i in range(0, seq_len, step)],
                       fontsize=7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight (Last Layer, Aggregated)', fontsize=9, fontweight='bold')

    # Plot 2: Information Density with cluster regions
    positions = np.arange(len(densities))

    # Color the background by cluster
    prev_boundary = 0
    for i, boundary in enumerate(boundary_positions[:-1]):
        if boundary < seq_len:
            ax_density.axvspan(prev_boundary, boundary,
                             alpha=0.3, color=colors[i % len(colors)])
            prev_boundary = boundary

    ax_density.plot(positions, densities, color='#2C3E50', linewidth=2.5)

    # Mark boundaries
    for boundary in boundary_positions:
        if boundary < seq_len:
            ax_density.axvline(x=boundary, color='#E74C3C', linewidth=1.5,
                               linestyle='--', alpha=0.8)

    ax_density.set_xlabel('Token Position', fontsize=11, fontweight='bold')
    ax_density.set_ylabel('Information Density D(t)', fontsize=11, fontweight='bold')
    ax_density.set_title('Contextual Information Density by Semantic Slice', fontsize=12, fontweight='bold')
    ax_density.set_ylim(-0.1, 1.1)
    ax_density.grid(True, alpha=0.3, color='#7F8C8D')

    # Add cluster labels to density plot
    prev_boundary = 0
    for i, boundary in enumerate(boundary_positions[:-1]):
        if boundary < seq_len and (boundary - prev_boundary) >= 3:
            mid = (prev_boundary + boundary) / 2
            ax_density.text(mid, 0.85, f"σ{i+1}",
                          ha='center', va='center',
                          fontsize=8, fontweight='bold',
                          color=colors[i % len(colors)],
                          bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      alpha=0.8,
                                      edgecolor=colors[i % len(colors)]))
        prev_boundary = boundary

    # Plot 3: Legend
    ax_legend.axis('off')
    legend_text = "Semantic Clusters (σ):\n\n"
    for i in range(min(len(boundary_positions) - 1, len(colors))):
        legend_text += f"σ{i+1} = {colors[i]}\n"

    ax_legend.text(0.1, 0.5, legend_text,
                  transform=ax_legend.transAxes,
                  fontsize=10, verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                  family='monospace')

    ax_legend.axis('off')

    plt.tight_layout()
    return fig


def main():
    """Run the enhanced attention visualization."""
    print("=" * 70)
    print("AgentOS Figure 3.2: Attention Matrix with Semantic Clusters")
    print("=" * 70)
    print()
    print("This visualization reproduces Figure 3.2 from the AgentOS paper.")
    print("Notice the rectangular blocks along the diagonal - these are the")
    print("semantic clusters where tokens strongly attend to each other.")
    print()

    # Sample text designed to create clear semantic clusters
    sample_text = """
    The human brain has approximately 86 billion neurons. These neurons form
    complex networks that enable cognition and consciousness throughout the body.

    In contrast, artificial neural networks are mathematical models inspired by
    biological neurons. They use layers of interconnected nodes to process
    information through weighted connections learned from training data.

    The main difference is that biological neurons operate in parallel using
    electrochemical signals, while artificial neurons process data sequentially
    through matrix operations on digital computers.
    """.strip()

    print(f"Processing text ({len(sample_text)} chars)...")

    # Create kernel
    kernel = create_kernel(device="auto")
    kernel.load()

    # Process
    result = kernel.process(sample_text, max_length=128)

    attention = result.attention_output.attention_weights
    tokens = result.attention_output.tokens
    slicing = result.slicing_result

    print(f"Processed {len(tokens)} tokens")
    print(f"Found {len(slicing.slices)} semantic clusters/slices")
    print(f"Attention shape: {attention.shape}")
    print()

    # Generate visualization
    print("Generating cluster visualization...")
    fig = plot_attention_with_clusters(
        attention_weights=attention,
        tokens=tokens,
        boundary_positions=slicing.metadata['boundary_positions'],
        densities=slicing.density_profile.densities,
    )

    # Save
    output_path = Path(__file__).parent / "attention_clusters.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    output_hd = Path(__file__).parent / "attention_clusters_hd.png"
    fig.savefig(output_hd, dpi=300, bbox_inches='tight')
    print(f"✓ Saved HD: {output_hd}")

    # Print cluster details
    print()
    print("Semantic Clusters Detected:")
    print("-" * 70)
    for i, slice_ in enumerate(slicing.slices):
        preview = slice_.content[:60].replace('\n', ' ')
        print(f"σ{i+1:2d} | [{slice_.token_count:3d} tokens] | D={slice_.density_mean:.3f} | {preview}...")

    kernel.unload()

    print()
    print("=" * 70)
    print("The visualization should show clear rectangular blocks along the diagonal.")
    print("Each colored block represents a semantic cluster (slice σ).")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
