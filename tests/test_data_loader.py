#!/usr/bin/env python3
"""Simple test script for the data_loader helpers.

This is a small manual test / exploration tool rather than a unit test.
It loads conversations from JSON files in the project's `data/` directory
and prints some basic statistics.
"""

import sys
from pathlib import Path

# Add project root to path so we can import eval_app
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval_app import load_all_conversations, get_all_headers


def main() -> int:
    # Get data directory from project root
    data_dir = project_root / "data"

    print(f"Loading conversations from: {data_dir}")
    print("-" * 60)

    # Load all conversations
    conversations = load_all_conversations(data_dir)

    if not conversations:
        print("No conversations loaded. Check the data directory and file format.")
        return 1

    print(f"\n✓ Successfully loaded {len(conversations)} conversation(s)\n")

    # Show all available headers
    headers = get_all_headers(conversations)
    print(f"Available headers ({len(headers)}):")
    for header in headers:
        print(f"  - {header}")

    # Show details of first conversation
    if conversations:
        print("\n" + "=" * 60)
        print("First Conversation Details:")
        print("=" * 60)

        first_conv = conversations[0]
        for key, value in first_conv.items():
            if key == "conversation":
                # Truncate long conversation text
                preview = value[:300] + "..." if len(value) > 300 else value
                print(f"\n{key}:")
                print(f"  {preview}")
            else:
                print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual script
    sys.exit(main())
