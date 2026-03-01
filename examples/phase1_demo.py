#!/usr/bin/env python3
"""
Phase 1 Demo: Reasoning Kernel & Semantic Slicing

This demo demonstrates:
1. Loading Qwen2.5-0.5B on Mac M1 with MPS
2. Processing text through the Reasoning Kernel
3. Extracting attention weights
4. Computing Contextual Information Density (CID)
5. Detecting semantic boundaries
6. Creating semantic slices

Usage:
    python examples/phase1_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentos import create_kernel
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    """Run the Phase 1 demo."""
    rprint(Panel.fit("[bold cyan]AgentOS Phase 1 Demo[/bold cyan]"))
    rprint()

    # Sample text with multiple topics for semantic slicing
    sample_text = """
    The human brain is composed of approximately 86 billion neurons.
    These neurons communicate through electrical and chemical signals,
    forming complex neural networks that enable cognition and consciousness.

    In contrast, modern artificial neural networks are based on mathematical
    abstractions of biological neurons. They use layers of interconnected
    nodes that process information through weighted connections.

    The key difference lies in the fundamental architecture: biological
    neurons operate in a massively parallel, analog fashion, while
    artificial neurons typically process data in discrete, sequential layers.
    """.strip()

    rprint(Panel(sample_text, title="[bold]Input Text[/bold]", border_style="blue"))
    rprint()

    # Create Reasoning Kernel
    rprint("[yellow]Loading Qwen2.5-0.5B model...[/yellow]")
    rprint("[dim](This will download ~1GB on first run)[/dim]")

    kernel = create_kernel(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="auto",  # Will detect MPS on Mac
        l1_max_tokens=512,
    )

    # Load the model
    kernel.load()
    rprint("[green]✓ Model loaded[/green]")
    rprint()

    # Show kernel info
    info = kernel.get_context_summary()
    rprint(Panel(
        f"""Device: {info['device']}
Model: {info['model']}
State: {info['kernel_state']}""",
        title="[bold]Kernel Info[/bold]",
        border_style="green",
    ))
    rprint()

    # Process the text
    rprint("[yellow]Processing text through Reasoning Kernel...[/yellow]")
    result = kernel.process(sample_text, max_length=512)
    rprint("[green]✓ Processing complete[/green]")
    rprint()

    # Show attention output info
    rprint(Panel(
        f"""Sequence length: {len(result.attention_output.tokens)}
Layers: {result.attention_output.metadata['num_layers']}
Heads: {result.attention_output.metadata['num_heads']}
Attention shape: {result.attention_output.attention_weights.shape}""",
        title="[bold]Attention Output[/bold]",
        border_style="cyan",
    ))
    rprint()

    # Show slicing results
    rprint(f"[yellow]Found {len(result.slicing_result.slices)} semantic slices[/yellow]")
    rprint()

    # Create table for slices
    table = Table(title="Semantic Slices", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Tokens")
    table.add_column("Density", justify="right")
    table.add_column("Importance", justify="right")
    table.add_column("Content Preview")

    for i, slice_ in enumerate(result.slicing_result.slices):
        # Truncate content for display
        preview = slice_.content[:60] + "..." if len(slice_.content) > 60 else slice_.content

        table.add_row(
            f"{i+1}",
            str(slice_.token_count),
            f"{slice_.density_mean:.3f}",
            f"{slice_.importance_score:.3f}",
            preview,
        )

    console.print(table)
    rprint()

    # Show density profile statistics
    stats = result.slicing_result.get_slice_statistics()
    rprint(Panel(
        f"""Total slices: {stats['count']}
Mean tokens per slice: {stats['mean_tokens']:.1f} (±{stats['std_tokens']:.1f})
Mean density: {stats['mean_density']:.3f} (±{stats['std_density']:.3f})""",
        title="[bold]Slice Statistics[/bold]",
        border_style="blue",
    ))
    rprint()

    # Show boundary detection info
    metadata = result.slicing_result.metadata
    rprint(Panel(
        f"""Boundary threshold ε: {metadata.get('boundary_threshold', 'N/A'):.4f}
Boundaries detected: {metadata.get('num_boundaries', 'N/A')}
Boundary positions: {metadata.get('boundary_positions', [])[:10]}...""",
        title="[bold]Boundary Detection[/bold]",
        border_style="yellow",
    ))
    rprint()

    # Show cognitive state
    state = result.resulting_state
    rprint(Panel(
        f"""Active slices: {len(state.active_slices)}
Active tokens: {state.active_token_count}
Attention focus: {state.attention_focus:.3f}""",
        title="[bold]Cognitive State[/bold]",
        border_style="green",
    ))
    rprint()

    # Unload to free memory
    rprint("[yellow]Unloading model...[/yellow]")
    kernel.unload()
    rprint("[green]✓ Unloaded[/green]")

    rprint()
    rprint(Panel.fit("[bold green]Demo complete![/bold green]"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]Demo interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        rprint(f"\n[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)
