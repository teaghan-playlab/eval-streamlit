#!/usr/bin/env python3
"""
Simple test script for data_loader.py
Loads conversations from JSON files in the data directory.
"""

import sys
from pathlib import Path
from data_loader import load_all_conversations, get_all_headers


def main():
    # Get data directory (relative to this script)
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
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
    
    # Show summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    
    if conversations:
        # Count by app
        app_counts = {}
        for conv in conversations:
            app_name = conv.get("appName", "Unknown")
            app_counts[app_name] = app_counts.get(app_name, 0) + 1
        
        print(f"\nConversations by App:")
        for app_name, count in sorted(app_counts.items()):
            print(f"  {app_name}: {count}")
        
        # Average message counts
        avg_user_msgs = sum(c.get("userMessageCount", 0) for c in conversations) / len(conversations)
        avg_total_msgs = sum(c.get("messageCount", 0) for c in conversations) / len(conversations)
        
        print(f"\nAverage user messages per conversation: {avg_user_msgs:.1f}")
        print(f"Average total messages per conversation: {avg_total_msgs:.1f}")
        
        # Show custom headers found
        custom_headers = []
        standard_headers = {
            "appId", "appName", "conversationId", "modelName", "userId",
            "createdAt", "endedAt", "userMessageCount", "messageCount",
            "flaggedMessageCount", "moderatedMessageCount", "conversation"
        }
        
        for conv in conversations:
            for key in conv.keys():
                if key not in standard_headers and key not in custom_headers:
                    custom_headers.append(key)
        
        if custom_headers:
            print(f"\nCustom headers extracted from user-start messages:")
            for header in sorted(custom_headers):
                print(f"  - {header}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
