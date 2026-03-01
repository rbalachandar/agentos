#!/usr/bin/env python3
"""
Phase 2 Demo: Cognitive Memory Hierarchy (CMH)

Demonstrates:
- L1 Cache (Active Attention Window)
- L2 RAM (Deep Context)
- L3 Storage (Knowledge Base)
- S-MMU (Semantic Memory Management Unit)
- Importance-based paging
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from agentos import create_kernel
from agentos.memory import SMMU, SMMUConfig, L1CacheConfig, L2Config, L3Config


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Demonstrate Phase 2 functionality."""
    print_section("AgentOS Phase 2: Cognitive Memory Hierarchy Demo")

    # Create a small Reasoning Kernel for slicing
    print("\n1. Initializing Reasoning Kernel...")
    kernel = create_kernel(device="auto")
    kernel.load()

    # Sample text with multiple semantic sections
    sample_text = """
    The human brain has approximately 86 billion neurons organized into
    complex networks. These neurons communicate through electrochemical signals
    across synapses, enabling cognition and consciousness.

    In contrast, artificial neural networks are computational models inspired
    by biological neurons. They use mathematical functions and weighted
    connections to process information through layers.

    The key difference lies in architecture: biological neurons operate in
    parallel with analog signals, while artificial neurons process data
    sequentially using digital computations.
    """.strip()

    print(f"Processing text ({len(sample_text)} chars)...")

    # Process and get semantic slices
    result = kernel.process(sample_text, max_length=128)
    slices = result.slicing_result.slices
    hidden_states = result.attention_output.hidden_states

    print(f"Generated {len(slices)} semantic slices")
    print()

    # Initialize S-MMU with small L1 to demonstrate paging
    print("2. Initializing S-MMU (Semantic Memory Management Unit)...")
    smmu_config = SMMUConfig(
        l1_config=L1CacheConfig(
            max_tokens=100,  # Very small to force paging
            max_slices=5,
        ),
        l2_config=L2Config(
            max_tokens=1000,
            max_slices=50,
        ),
        l3_config=L3Config(
            storage_path="./data/l3_demo",
        ),
    )
    smmu = SMMU(smmu_config)

    print(f"L1 Cache: {smmu.config.l1_config.max_tokens} tokens max")
    print(f"L2 RAM: {smmu.config.l2_config.max_tokens} tokens max")
    print(f"L3 Storage: {smmu.config.l3_config.storage_path}")
    print()

    # Process slices through S-MMU
    print_section("3. Processing Slices through S-MMU")
    l1_slice_ids = smmu.process_slices(result.slicing_result, hidden_states)

    print(f"Slices promoted to L1: {len(l1_slice_ids)}")
    for i, slice_id in enumerate(l1_slice_ids):
        slice_data = smmu.l1.get(slice_id)
        if slice_data:
            print(f"  L1[{i}]: {slice_data.slice_data.content[:50]}...")
    print()

    # Show memory statistics
    print_section("4. Memory Statistics")
    stats = smmu.get_memory_stats()

    print("L1 Cache:")
    print(f"  Utilization: {stats['l1']['utilization']:.1%}")
    print(f"  Tokens: {stats['l1']['used_tokens']}/{stats['l1']['max_tokens']}")
    print(f"  Slices: {stats['l1']['slice_count']}")
    print()

    print("L2 RAM:")
    print(f"  Utilization: {stats['l2']['utilization']:.1%}")
    print(f"  Tokens: {stats['l2']['used_tokens']}/{stats['l2']['max_tokens']}")
    print(f"  Slices: {stats['l2']['slice_count']}")
    print()

    print("L3 Storage:")
    print(f"  Slices: {stats['l3']['slice_count']}")
    print(f"  Size: {stats['l3']['total_size_bytes']} bytes")
    print()

    print("Page Table:")
    print(f"  Total entries: {stats['page_table']['total']}")
    print(f"  L1 entries: {stats['page_table']['l1']}")
    print(f"  L2 entries: {stats['page_table']['l2']}")
    print(f"  L3 entries: {stats['page_table']['l3']}")
    print()

    # Demonstrate importance scores
    print_section("5. Slice Importance Scores")
    for i, slice_ in enumerate(slices):
        entry = smmu.page_table.get(slice_.id)
        if entry:
            print(f"σ{i+1}: ℐ={entry.importance_score:.3f} | {entry.tier.value} | {slice_.content[:40]}...")
    print()

    # Demonstrate slice retrieval
    print_section("6. Slice Retrieval (L1 → L2 Promotion)")
    for i, slice_ in enumerate(slices):
        slice_id = slice_.id
        retrieved = smmu.get_slice(slice_id)
        if retrieved:
            entry = smmu.page_table.get(slice_id)
            tier = entry.tier.value if entry else "Unknown"
            print(f"Retrieved σ{i+1} from {tier}: {retrieved.content[:40]}...")
    print()

    # Final statistics
    print_section("7. Final Memory Statistics")
    stats = smmu.get_memory_stats()

    print("L1 Cache (after retrieval):")
    print(f"  Utilization: {stats['l1']['utilization']:.1%}")
    print(f"  Tokens: {stats['l1']['used_tokens']}/{stats['l1']['max_tokens']}")
    print(f"  Slices: {stats['l1']['slice_count']}")
    print()

    print("Page Table (after retrieval):")
    print(f"  L1 entries: {stats['page_table']['l1']}")
    print(f"  L2 entries: {stats['page_table']['l2']}")
    print(f"  L3 entries: {stats['page_table']['l3']}")
    print()

    kernel.unload()

    print_section("Demo Complete!")
    print("The S-MMU successfully managed semantic slices across the memory hierarchy,")
    print("promoting important slices to L1 and paging out less important ones to L2/L3.")
    print()


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
