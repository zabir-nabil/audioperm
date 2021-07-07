"""
Helper functions for audioperm.
"""

def type_nested(iterable, tp):
    """
    Args:
        iterable (list): a list
        tp (type): type of iterable
    Returns:
        bool: If all are of same type.
    """
    if iterable == []:
        return False
    return all(isinstance(item, tp) for item in iterable)