"""Sandbox utilities."""

from typing import Set


def get_default_allowed_modules() -> Set[str]:
    """Get the default set of allowed modules for sandboxing.
    
    Returns:
        Set of module names
    """
    return {
        'json',
        're',
        'math',
        'random',
        'datetime',
        'collections',
        'itertools',
        'statistics',
        'typing',
        'string',
        'hashlib',
        'base64',
        'binascii',
        'decimal',
        'fractions',
        'numbers',
        'inspect',
        'textwrap',
        'difflib',
    }


def get_default_blocked_builtins() -> Set[str]:
    """Get the default set of blocked builtins for sandboxing.
    
    Returns:
        Set of builtin names
    """
    return {
        '__import__',
        'eval',
        'compile',
        'exec',
        'open',
        'exit',
        'quit',
        'input',
        'raw_input',  # Python 2 compatibility
    }
