#!/usr/bin/env python
"""
Apply compatibility fixes for using BeatNet with NumPy 1.20+

This script applies the necessary fixes to the madmom library to make
it compatible with newer versions of NumPy (1.20+) which have deprecated
numpy.float and numpy.int types.

Usage:
    python apply_fixes.py

The script will:
1. Find the installed madmom library
2. Apply the necessary patches to fix NumPy compatibility issues
3. Report success or failure

Author: BeatNet Contributors
"""

import os
import sys
import re
import importlib.util
import glob
from pathlib import Path

def find_madmom_path():
    """Find the installed madmom library path."""
    try:
        spec = importlib.util.find_spec('madmom')
        if spec is None:
            return None
        return os.path.dirname(spec.origin)
    except ImportError:
        return None

def apply_patches(madmom_path):
    """Apply patches to the madmom library."""
    if not madmom_path:
        print("Error: Could not find the madmom library.")
        return False
    
    patches = [
        # File: madmom/io/__init__.py
        {
            'file': os.path.join(madmom_path, 'io', '__init__.py'),
            'patterns': [
                (r"SEGMENT_DTYPE\s*=\s*\[\('start',\s*np\.float\)", 
                 "SEGMENT_DTYPE = [('start', np.float64)"),
                (r"\('end',\s*np\.float\)", 
                 "('end', np.float64)"),
            ]
        },
        # File: madmom/evaluation/chords.py
        {
            'file': os.path.join(madmom_path, 'evaluation', 'chords.py'),
            'patterns': [
                (r"CHORD_DTYPE\s*=\s*\[\('root',\s*np\.int\)", 
                 "CHORD_DTYPE = [('root', np.int32)"),
                (r"\('bass',\s*np\.int\)", 
                 "('bass', np.int32)"),
                (r"CHORD_ANN_DTYPE\s*=\s*\[\('start',\s*np\.float\)", 
                 "CHORD_ANN_DTYPE = [('start', np.float64)"),
                (r"\('end',\s*np\.float\)", 
                 "('end', np.float64)"),
                (r"NO_CHORD\s*=\s*\(-1,\s*-1,\s*np\.zeros\(12,\s*dtype=np\.int\)\)", 
                 "NO_CHORD = (-1, -1, np.zeros(12, dtype=np.int32))"),
                (r"UNKNOWN_CHORD\s*=\s*\(-1,\s*-1,\s*np\.ones\(12,\s*dtype=np\.int\)\s*\*\s*-1\)", 
                 "UNKNOWN_CHORD = (-1, -1, np.ones(12, dtype=np.int32) * -1)"),
                (r"\.astype\(np\.float\)", 
                 ".astype(np.float64)"),
            ]
        },
        # File: madmom/features/beats_hmm.py
        {
            'file': os.path.join(madmom_path, 'features', 'beats_hmm.py'),
            'patterns': [
                (r"self\.intervals\s*=\s*np\.ascontiguousarray\(intervals,\s*dtype=np\.int\)", 
                 "self.intervals = np.ascontiguousarray(intervals, dtype=np.int32)"),
                (r"self\.first_states\s*=\s*first_states\.astype\(np\.int\)", 
                 "self.first_states = first_states.astype(np.int32)"),
                (r"self\.state_intervals\s*=\s*np\.empty\(self\.num_states,\s*dtype=np\.int\)", 
                 "self.state_intervals = np.empty(self.num_states, dtype=np.int32)"),
                (r"self\.state_intervals\s*=\s*np\.empty\(0,\s*dtype=np\.int\)", 
                 "self.state_intervals = np.empty(0, dtype=np.int32)"),
                (r"self\.state_phases\s*=\s*np\.empty\(0,\s*dtype=np\.int\)", 
                 "self.state_phases = np.empty(0, dtype=np.int32)"),
                (r"np\.ones\(num_states,\s*dtype=np\.int\)", 
                 "np.ones(num_states, dtype=np.int32)"),
                (r"probabilities\s*=\s*np\.ones_like\(states,\s*dtype=np\.float\)", 
                 "probabilities = np.ones_like(states, dtype=np.float64)"),
                (r"log_densities\s*=\s*np\.empty\(\(len\(observations\),\s*[0-9]+\),\s*dtype=np\.float\)",
                 lambda m: m.group(0).replace("np.float", "np.float64")),
                (r"ratio\s*=\s*\(to_intervals\.astype\(np\.float\)",
                 "ratio = (to_intervals.astype(np.float64)")
            ]
        },
    ]
    
    success = True
    for patch in patches:
        file_path = patch['file']
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            success = False
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        for pattern, replacement in patch['patterns']:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Updated {os.path.basename(file_path)}")
        else:
            print(f"✓ No changes needed in {os.path.basename(file_path)} (already patched or not applicable)")
    
    return success

def add_monkey_patch_to_init(madmom_path):
    """Add monkey patch to madmom __init__.py to ensure compatibility."""
    init_file = os.path.join(madmom_path, '__init__.py')
    if not os.path.exists(init_file):
        print(f"Warning: Could not find madmom __init__.py at {init_file}")
        return False
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Don't add the patch if it's already there
    if "# NumPy 1.20+ compatibility" in content:
        print("✓ Monkey patch already present in madmom/__init__.py")
        return True
    
    patch = """
# NumPy 1.20+ compatibility
try:
    import numpy as np
    # Add compatibility for deprecated numpy types
    if not hasattr(np, 'float'):
        np.float = np.float64
    if not hasattr(np, 'int'):
        np.int = np.int32
except:
    pass
"""
    
    # Add the patch after the imports
    import_pattern = r"((?:from [^\n]+\n|import [^\n]+\n)+)"
    if re.search(import_pattern, content):
        content = re.sub(import_pattern, r"\1\n" + patch, content, count=1)
    else:
        # If no imports found, add at the beginning
        content = patch + content
    
    with open(init_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Added monkey patch to madmom/__init__.py")
    return True

def main():
    """Main function to apply patches."""
    print("BeatNet NumPy 1.20+ Compatibility Fixer")
    print("=======================================")
    
    madmom_path = find_madmom_path()
    if not madmom_path:
        print("Error: Could not find the madmom library. Please make sure it's installed.")
        return 1
    
    print(f"Found madmom library at: {madmom_path}")
    
    # Apply patches
    print("\nApplying patches...")
    success = apply_patches(madmom_path)
    
    # Add monkey patch
    print("\nAdding monkey patch to __init__.py...")
    success = add_monkey_patch_to_init(madmom_path) and success
    
    if success:
        print("\n✓ Successfully applied compatibility fixes!")
        print("\nYou should now be able to use BeatNet with NumPy 1.20+")
    else:
        print("\n⚠ Some patches could not be applied. Please check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 