import hashlib
from pathlib import Path

from knowlang.configs import DBConfig
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file contents.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Hexadecimal string representation of the SHA-256 hash
        
    Raises:
        IOError: If the file cannot be read
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        LOG.error(f"Error computing hash for {file_path}: {e}")
        raise

def get_relative_path(file_path: Path, db_config: DBConfig) -> Path:
    """Convert an absolute path to a path relative to the codebase directory.
    
    Args:
        file_path: The absolute path to convert
        db_config: Database configuration containing codebase_directory
        
    Returns:
        Path object representing the relative path
    """
    try:
        # Try to make path relative to the codebase directory
        return file_path.relative_to(db_config.codebase_directory)
    except ValueError:
        # If the path is not a subdirectory of codebase_directory,
        # return the original path as a fallback
        LOG.warning(f"Path {file_path} is not relative to codebase directory {db_config.codebase_directory}")
        return file_path

def get_absolute_path(relative_path: Path, db_config: DBConfig) -> Path:
    """Convert a relative path to an absolute path in the codebase directory.
    
    Args:
        relative_path: The relative path to convert
        db_config: Database configuration containing codebase_directory
        
    Returns:
        Path object representing the absolute path
    """
    return db_config.codebase_directory / relative_path