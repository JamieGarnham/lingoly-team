#!/usr/bin/env python3
import re
import random
from typing import List, Tuple

def shuffle_lines_between_tokens(text: str, start_token: str, end_token: str) -> str:
    """
    Shuffles lines between start and end tokens while preserving the tokens themselves.
    """
    pattern = f"({re.escape(start_token)})(.*?)({re.escape(end_token)})"
    
    def shuffle_section(match):
        start_tag = match.group(1)
        content = match.group(2)
        end_tag = match.group(3)
        
        # Split content into lines and filter out empty lines
        lines = [line for line in content.split('\n') if line.strip()]
        
        # Shuffle the lines
        random.shuffle(lines)
        
        # Rejoin with newlines, maintaining the original formatting
        shuffled_content = '\n' + '\n'.join(lines) + '\n'
        
        return start_tag + shuffled_content + end_tag
    
    return re.sub(pattern, shuffle_section, text, flags=re.DOTALL)


def reorder_numbered_lines(text: str, start_token: str, end_token: str) -> str:
    """
    Reorders numbered/lettered lines between start and end tokens, 
    updating the numbers/letters to maintain sequence.
    """
    pattern = f"({re.escape(start_token)})(.*?)({re.escape(end_token)})"
    
    def reorder_section(match):
        start_tag = match.group(1)
        content = match.group(2)
        end_tag = match.group(3)
        
        # Split content into lines and filter out empty lines
        lines = [line for line in content.split('\n') if line.strip()]
        
        if not lines:
            return match.group(0)
        
        # Parse lines to extract numbering format and content
        line_data = []
        numbering_format = None
        
        for line in lines:
            line = line.strip()
            
            # Check for numeric format (1, 2, 3, ...)
            numeric_match = re.match(r'^(\d+)\.?\s*(.*)', line)
            if numeric_match:
                if numbering_format is None:
                    numbering_format = 'numeric'
                elif numbering_format != 'numeric':
                    # Mixed formats, fall back to simple shuffling
                    random.shuffle(lines)
                    shuffled_content = '\n' + '\n'.join(lines) + '\n'
                    return start_tag + shuffled_content + end_tag
                
                line_data.append((int(numeric_match.group(1)), numeric_match.group(2)))
                continue
            
            # Check for alphabetic format (a, b, c, ...)
            alpha_match = re.match(r'^([a-z])\.?\s*(.*)', line)
            if alpha_match:
                if numbering_format is None:
                    numbering_format = 'alphabetic'
                elif numbering_format != 'alphabetic':
                    # Mixed formats, fall back to simple shuffling
                    random.shuffle(lines)
                    shuffled_content = '\n' + '\n'.join(lines) + '\n'
                    return start_tag + shuffled_content + end_tag
                
                letter_value = ord(alpha_match.group(1)) - ord('a') + 1
                line_data.append((letter_value, alpha_match.group(2)))
                continue
            
            # Line doesn't match expected format, fall back to simple shuffling
            random.shuffle(lines)
            shuffled_content = '\n' + '\n'.join(lines) + '\n'
            return start_tag + shuffled_content + end_tag
        
        # If we get here, all lines matched the same format
        if not line_data:
            return match.group(0)
        
        # Extract just the content (without numbers/letters) and shuffle
        contents = [data[1] for data in line_data]
        random.shuffle(contents)
        
        # Reassemble with new numbering
        new_lines = []
        for i, content in enumerate(contents):
            if numbering_format == 'numeric':
                new_lines.append(f"{i + 1}. {content}")
            elif numbering_format == 'alphabetic':
                letter = chr(ord('a') + i)
                new_lines.append(f"{letter}. {content}")
        
        # Rejoin with newlines, maintaining the original formatting
        reordered_content = '\n' + '\n'.join(new_lines) + '\n'
        
        return start_tag + reordered_content + end_tag
    
    return re.sub(pattern, reorder_section, text, flags=re.DOTALL)


def remove_shuffle_tokens(text: str) -> str:
    """
    Removes all shuffle and reorder tokens from the text.
    """
    tokens_to_remove = [
        '<SHUFFLE_START>',
        '<SHUFFLE_END>',
        '<REORDER_START>',
        '<REORDER_END>'
    ]
    
    result = text
    for token in tokens_to_remove:
        result = result.replace(token, '')
    
    # Clean up any extra newlines that might have been left
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result


def shuffle_context(context: str) -> str:
    """
    Main function to shuffle context by processing all shuffle and reorder sections,
    then removing the tokens.
    """
    # First, shuffle SHUFFLE_START/END sections
    shuffled = shuffle_lines_between_tokens(context, '<SHUFFLE_START>', '<SHUFFLE_END>')
    
    # Then, reorder REORDER_START/END sections
    reordered = reorder_numbered_lines(shuffled, '<REORDER_START>', '<REORDER_END>')
    
    # Finally, remove all tokens
    clean = remove_shuffle_tokens(reordered)
    
    return clean


def has_shuffle_tokens(context: str) -> bool:
    """
    Check if the context contains any shuffle or reorder tokens.
    """
    tokens = ['<SHUFFLE_START>', '<REORDER_START>']
    return any(token in context for token in tokens)


# Test the functions if run directly
if __name__ == "__main__":
    # Test shuffle function
    test_shuffle_text = """Some text before.
<SHUFFLE_START>
 bimöştü         wrestler
 beddi         lookout, optician  
 üpgontüd         unsuccessful
 ütürtüd         nameless
<SHUFFLE_END>
Some text after."""

    print("Original shuffle text:")
    print(test_shuffle_text)
    print("\nShuffled:")
    print(shuffle_context(test_shuffle_text))
    
    # Test reorder function
    test_reorder_text = """Some text before.
<REORDER_START>
1. First item
2. Second item  
3. Third item
4. Fourth item
<REORDER_END>
Some text after."""

    print("\n\nOriginal reorder text:")
    print(test_reorder_text)
    print("\nReordered:")
    print(shuffle_context(test_reorder_text))