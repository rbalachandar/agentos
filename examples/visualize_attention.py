#!/usr/bin/env python3
"""
Reproduce Figure 3.2 from AgentOS paper: Attention Matrix Heatmap.

This script demonstrates:
1. Extracting attention weights from Qwen2.5-0.5B
2. Computing Contextual Information Density (CID)
3. Detecting semantic slice boundaries
4. Visualizing the attention matrix with slice boundaries overlaid

The heatmap shows how attention forms "block structures" that correspond
to semantic slices - tokens within a slice attend strongly to each other.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Rectangle

from agentos import create_kernel


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: list[str],
    boundary_positions: list[int],
    densities: np.ndarray,
    title: str = "AgentOS Attention Matrix with Semantic Slices",
) -> plt.Figure:
    """Plot attention heatmap with semantic slice boundaries.

    Args:
        attention_weights: (num_layers, num_heads, seq_len, seq_len) attention weights
        tokens: List of tokens
        boundary_positions: List of semantic boundary positions
        densities: (seq_len,) density values
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    # Aggregate attention across layers and heads (mean)
    # Shape: (seq_len, seq_len)
    aggregated_attention = attention_weights.mean(axis=(0, 1))

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1]},
    )

    # Plot 1: Attention heatmap with boundaries
    im = ax1.imshow(aggregated_attention, cmap='viridis', aspect='auto', origin='upper')

    # Add semantic slice boundaries as horizontal and vertical lines
    for boundary in boundary_positions:
        if boundary < len(tokens):
            # Horizontal line (boundary in query)
            ax1.axhline(y=boundary, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
            # Vertical line (boundary in key)
            ax1.axvline(x=boundary, color='white', linewidth=1.5, linestyle='--', alpha=0.7)

    # Configure axes
    ax1.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Query Position', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Show every nth token to avoid overcrowding
    n_tokens = len(tokens)
    step = max(1, n_tokens // 15)

    # Position ticks
    ax1.set_xticks(range(0, n_tokens, step))
    ax1.set_yticks(range(0, n_tokens, step))
    ax1.set_xticklabels([tokens[i] if i < n_tokens else str(i) for i in range(0, n_tokens, step)],
                       rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels([tokens[i] if i < n_tokens else str(i) for i in range(0, n_tokens, step)],
                       fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Attention Weight (Aggregated)', fontsize=10, fontweight='bold')

    # Plot 2: Information Density profile
    positions = np.arange(len(densities))
    ax2.plot(positions, densities, color='#2ecc71', linewidth=2, label='D(t)')
    ax2.fill_between(positions, 0, densities, alpha=0.3, color='#2ecc71')

    # Mark boundaries on density plot
    for boundary in boundary_positions:
        if boundary < len(densities):
            ax2.axvline(x=boundary, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7)

    ax2.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Information Density D(t)', fontsize=12, fontweight='bold')
    ax2.set_title('Contextual Information Density', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    # Add vertical span annotations for slices
    prev_boundary = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(boundary_positions)))
    for i, boundary in enumerate(boundary_positions):
        if boundary < len(densities) and boundary > prev_boundary:
            # Add slice label
            slice_center = (prev_boundary + boundary) / 2
            ax2.text(
                slice_center, 0.95,
                f"σ{i+1}",
                ha='center',
                va='top',
                fontsize=9,
                fontweight='bold',
                color=colors[i % len(colors)],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            )
        prev_boundary = boundary

    plt.tight_layout()
    return fig


def main():
    """Run the attention visualization demo."""
    print("=" * 60)
    print("AgentOS Figure 3.2: Attention Matrix Heatmap with Semantic Slices")
    print("=" * 60)
    print()

    # Sample text with clear semantic structure
    # Similar to paper example: has multiple topics that should form distinct slices
    sample_text = """
    The solar system is our cosmic neighborhood. At its center shines the Sun,
    a massive star that provides light and warmth to all orbiting planets.

    Mercury is the closest planet to the Sun and has extreme temperature variations.
    Venus has a thick atmosphere and is the hottest planet in our solar system.

    Earth is our home planet, the only known world to harbor life. It has one
    natural satellite called the Moon that influences our oceans through tides.

    Beyond Earth lies the asteroid belt, followed by the gas giants Jupiter and Saturn.
    These outer planets are much larger than the inner planets and have numerous moons.
    """.strip()

    print(f"Processing text ({len(sample_text)} characters)...")
    print()

    # Create Reasoning Kernel
    print("Loading Qwen2.5-0.5B model...")
    kernel = create_kernel(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="auto",
    )
    kernel.load()

    # Process the text
    result = kernel.process(sample_text, max_length=512)

    # Extract data for visualization
    attention_weights = result.attention_output.attention_weights
    tokens = result.attention_output.tokens
    slicing_result = result.slicing_result
    densities = slicing_result.density_profile.densities
    boundaries = slicing_result.metadata['boundary_positions']

    print(f"Processed {len(tokens)} tokens")
    print(f"Found {len(slicing_result.slices)} semantic slices")
    print(f"Attention shape: {attention_weights.shape}")
    print()

    # Create visualization
    print("Generating attention heatmap...")
    fig = plot_attention_heatmap(
        attention_weights=attention_weights,
        tokens=tokens,
        boundary_positions=boundaries,
        densities=densities,
        title="AgentOS Attention Matrix: Semantic Slice Detection (Figure 3.2)",
    )

    # Save the figure
    output_path = Path(__file__).parent / "attention_heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {output_path}")

    # Also save a high-resolution version
    output_path_hd = Path(__file__).parent / "attention_heatmap_hd.png"
    fig.savefig(output_path_hd, dpi=300, bbox_inches='tight')
    print(f"✓ Saved HD heatmap to: {output_path_hd}")

    # Print slice information
    print()
    print("Semantic Slices Detected:")
    print("-" * 60)
    for i, slice_ in enumerate(slicing_result.slices):
        preview = slice_.content[:80].replace('\n', ' ')
        if len(preview) < len(slice_.content):
            preview += "..."
        print(f"σ{i+1:2d} | [{slice_.token_count:3d} tokens] | D={slice_.density_mean:.3f} | {preview}")

    print()

    # Clean up
    kernel.unload()
    print("Model unloaded")

    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
