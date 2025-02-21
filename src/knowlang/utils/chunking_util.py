from pathlib import Path

from knowlang.configs import DBConfig

MAX_CHARS_PER_CHUNK = 10000  # Approximate 8k tokens limit (very rough estimate)


def convert_to_relative_path(path: Path, db_config: DBConfig) -> str:
    return str(path.relative_to(db_config.codebase_directory).as_posix())

def format_code_summary(code: str, summary: str) -> str:
    """Format code and summary into a single string"""
    return f"CODE:\n{code}\n\nSUMMARY:\n{summary}"

def truncate_chunk(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    """Truncate text to approximate token limit while preserving structure"""
    if len(text) <= max_chars:
        return text
    
    # Split into CODE and SUMMARY sections
    parts = text.split("\nSUMMARY:\n")
    if len(parts) != 2:
        # If structure not found, just truncate
        return text[:max_chars]
    
    code, summary = parts
    
    # Calculate available space for each section (proportionally)
    total_len = len(code) + len(summary)
    code_ratio = len(code) / total_len
    
    # Allocate characters proportionally
    code_chars = int(max_chars * code_ratio)
    summary_chars = max_chars - code_chars
    
    truncated_code = code[:code_chars]
    truncated_summary = summary[:summary_chars]
    
    return f"{truncated_code}\nSUMMARY:\n{truncated_summary}"