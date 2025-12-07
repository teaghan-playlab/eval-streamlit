import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_user_start_message(content: str) -> Dict[str, str]:
    """
    Parse the user-start message to extract custom headers.
    
    Splits by "\n\n" then right-splits by ":**" to get header (left) and value (right).
    
    Args:
        content: The content of the user-start message
        
    Returns:
        Dictionary mapping header names to values
    """
    headers = {}
    
    # Split by double newlines to get individual question-answer pairs
    parts = content.split("\n\n")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Right split by ":**" to separate header from value
        if ":**" in part:
            header, value = part.rsplit(":**", 1)
            # Clean up header (remove markdown formatting if present)
            header = header.strip().replace("**", "").strip()
            value = value.strip()
            if header and value:
                headers[header] = value
    
    return headers


def format_conversation_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Organize messages (excluding user-start) into a formatted conversation string.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Formatted conversation string with "**Provider Message:**\n\n" or 
        "**User Message:**\n\n" prefixes
    """
    conversation_parts = []
    
    for message in messages:
        source = message.get("source", "")
        content = message.get("content", "").strip()
        
        # Skip user-start messages and empty messages
        if source == "user-start" or not content:
            continue
        
        # Determine prefix based on source
        if source == "provider":
            prefix = "**Provider Message:**\n\n"
        elif source == "user":
            prefix = "**User Message:**\n\n"
        else:
            # Handle other sources (e.g., "user-start" already filtered)
            prefix = f"**{source.title()} Message:**\n\n"
        
        conversation_parts.append(prefix + content)
    
    return "\n\n".join(conversation_parts)


def load_conversations_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load conversations from a single JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of conversation dictionaries with extracted fields
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON file {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []
    
    conversations = data.get("conversations", [])
    if not conversations:
        logging.warning(f"No conversations found in {file_path}")
        return []
    
    processed_conversations = []
    
    for conv in conversations:
        try:
            # Extract top-level fields
            processed_conv = {
                "appId": conv.get("appId", ""),
                "appName": conv.get("appName", ""),
                "conversationId": conv.get("conversationId", ""),
                "chatbotModel": conv.get("modelName", ""),  # Rename modelName to chatbotModel
                "userId": conv.get("userId", ""),
                "createdAt": conv.get("createdAt", ""),
                "endedAt": conv.get("endedAt", ""),
            }
            
            # Extract conversation metrics
            metrics = conv.get("conversationMetric", {})
            processed_conv.update({
                "userMessageCount": metrics.get("userMessageCount", 0),
                "messageCount": metrics.get("messageCount", 0),
                "flaggedMessageCount": metrics.get("flaggedMessageCount", 0),
                "moderatedMessageCount": metrics.get("moderatedMessageCount", 0),
            })
            
            # Process messages
            messages = conv.get("messages", [])
            
            # Find and parse user-start message
            user_start_content = None
            for msg in messages:
                if msg.get("source") == "user-start":
                    user_start_content = msg.get("content", "")
                    break
            
            # Extract custom headers from user-start message
            if user_start_content:
                custom_headers = parse_user_start_message(user_start_content)
                processed_conv.update(custom_headers)
            
            # Format conversation (excluding user-start messages)
            conversation_text = format_conversation_messages(messages)
            
            # Prepend user-start content to conversation if it exists
            if user_start_content:
                user_start_prefixed = "**User Starter Inputs:**\n\n" + user_start_content.strip()
                if conversation_text:
                    processed_conv["conversation"] = user_start_prefixed + "\n\n" + conversation_text
                else:
                    processed_conv["conversation"] = user_start_prefixed
            else:
                processed_conv["conversation"] = conversation_text
            
            processed_conversations.append(processed_conv)
            
        except Exception as e:
            logging.error(f"Error processing conversation {conv.get('conversationId', 'unknown')}: {e}")
            continue
    
    return processed_conversations


def load_all_conversations(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all conversations from JSON files in the specified directory.
    
    Args:
        data_dir: Path to directory containing JSON files
        
    Returns:
        List of all conversation dictionaries from all files
    """
    if not data_dir.exists():
        logging.error(f"Data directory does not exist: {data_dir}")
        return []
    
    all_conversations = []
    
    # Find all JSON files in the directory
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        logging.warning(f"No JSON files found in {data_dir}")
        return []
    
    logging.info(f"Found {len(json_files)} JSON file(s) to process")
    
    for json_file in json_files:
        logging.info(f"Processing {json_file.name}")
        conversations = load_conversations_from_file(json_file)
        all_conversations.extend(conversations)
        logging.info(f"Loaded {len(conversations)} conversation(s) from {json_file.name}")
    
    logging.info(f"Total conversations loaded: {len(all_conversations)}")
    return all_conversations


def get_all_headers(conversations: List[Dict[str, Any]]) -> List[str]:
    """
    Get all unique header names from conversations (including custom headers).
    
    Useful for determining CSV column headers.
    
    Args:
        conversations: List of conversation dictionaries
        
    Returns:
        Sorted list of all unique header names
    """
    headers = set()
    
    # Standard headers
    standard_headers = [
        "appId",
        "appName",
        "conversationId",
        "chatbotModel",
        "userId",
        "createdAt",
        "endedAt",
        "userMessageCount",
        "messageCount",
        "flaggedMessageCount",
        "moderatedMessageCount",
        "conversation",
    ]
    headers.update(standard_headers)
    
    # Collect custom headers from all conversations
    for conv in conversations:
        headers.update(conv.keys())
    
    return sorted(list(headers))


def order_csv_headers(headers: List[str]) -> List[str]:
    """
    Order CSV headers in a specific order:
    1. Metadata (IDs, dates, message counts)
    2. Evaluation fields (evaluation_*)
    3. Starter inputs (custom fields from user-start)
    4. Conversation text
    
    Args:
        headers: List of header names
        
    Returns:
        Ordered list of headers
    """
    # Define metadata fields in desired order (IDs, dates, counts)
    # Separate chatbot model and evaluator model
    metadata_order = [
        "appId",
        "appName",
        "conversationId",
        "createdAt",
        "endedAt",
        "userId",
        "chatbotModel",  # Chatbot model
        "evaluatorModel",  # Evaluator model
        "messageCount",
        "userMessageCount",
        "flaggedMessageCount",
        "moderatedMessageCount",
    ]
    
    # Separate headers into categories
    metadata = []
    evaluation_metadata = []  # evaluation_config_file, evaluation_decode_failed, etc.
    evaluation_fields = []  # evaluation_* fields that are actual evaluation results
    starter_inputs = []
    conversation_field = []
    other_fields = []
    
    # Known evaluation prefixes for actual evaluation results
    evaluation_prefixes = ["evaluation_"]
    
    # Evaluation metadata fields (not actual evaluation results)
    evaluation_metadata_fields = ["evaluation_config_file", "evaluation_decode_failed", "evaluation_error"]
    
    # Known metadata fields
    metadata_set = set(metadata_order)
    
    for header in headers:
        if header == "conversation":
            conversation_field.append(header)
        elif header in metadata_set:
            metadata.append(header)
        elif header in evaluation_metadata_fields:
            evaluation_metadata.append(header)
        elif any(header.startswith(prefix) for prefix in evaluation_prefixes):
            evaluation_fields.append(header)
        else:
            # Assume it's a starter input (custom field from user-start message)
            starter_inputs.append(header)
    
    # Order metadata according to metadata_order
    ordered_metadata = [h for h in metadata_order if h in metadata]
    # Add any metadata fields not in the predefined order
    ordered_metadata.extend([h for h in metadata if h not in metadata_order])
    
    # Sort evaluation metadata and evaluation fields
    evaluation_metadata.sort()
    evaluation_fields.sort()
    starter_inputs.sort()
    
    # Combine in desired order: metadata, evaluation_metadata, evaluation_fields, starter_inputs, conversation
    ordered_headers = ordered_metadata + evaluation_metadata + evaluation_fields + starter_inputs + conversation_field + other_fields
    
    return ordered_headers
