import numpy as np
from hashlib import blake2b
import json

def get_roots_hash_blake2b(roots, precision = 6, digest_size = 16):
    """
    Calculate a BLAKE2b hash from a list of complex roots.
    The order of roots should not affect the hash, so the roots are sorted.
    
    Parameters:
        roots (list of complex): List of complex roots.
        precision (int): Decimal places to round the real and imaginary parts.
        digest_size (int): The size of the digest in bytes (e.g., 16 for 128 bits).
        
    Returns:
        str: A BLAKE2b hash hexdigest representing the unique combination of roots.
    """
    # Standardize each root by rounding the real and imaginary parts
    standardized_roots = []
    for r in roots:
        real_str = format(round(r.real, precision), f'.{precision}f')
        imag_str = format(round(r.imag, precision), f'.{precision}f')
        standardized_roots.append(f"{real_str}_{imag_str}")
    
    # Sort the standardized strings to ensure order-independence
    standardized_roots.sort()
    
    # Concatenate into a single string
    hash_input = "_".join(standardized_roots)

    # Generate a BLAKE2b hash
    hash_object = blake2b(hash_input.encode('utf-8'), digest_size = digest_size)
    return hash_object.hexdigest()

def serialize_roots(roots):
    """
    Serialize a list of complex roots into a JSON string.

    This function converts the input into a NumPy array,
    sorts the complex numbers using np.sort_complex
    (which sorts by real part and then by imaginary part),
    and then serializes each complex number to a dictionary with 'real' and 'imag' keys.
    
    Parameters:
        roots (iterable of complex): An iterable of complex numbers representing the roots.
        
    Returns:
        str: A JSON-formatted string representing the sorted list of roots.
    """
    sorted_roots = np.sort_complex(np.array(roots))
    return json.dumps([{'real': r.real, 'imag': r.imag} for r in sorted_roots])

def as_complex(dct):
    """
    Convert a dictionary with 'real' and 'imag' keys into a complex number.

    This function takes a dictionary
    expected to have numeric values for keys 'real' and 'imag',
    and returns the corresponding complex number.

    Parameters:
        dct (dict): A dictionary with keys 'real' and 'imag'.

    Returns:
        complex: A complex number
            constructed from the dictionary's 'real' and 'imag' values.
    """
    return complex(dct['real'], dct['imag'])
