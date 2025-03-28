# NumPy 1.20+ Compatibility for BeatNet

## Issue

BeatNet relies on the `madmom` library, which uses deprecated NumPy types like `np.float` and `np.int` that were removed in NumPy 1.20. This causes errors when running BeatNet with modern versions of NumPy.

Common error messages include:
```
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself.
```
or
```
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself.
```

## Solutions

We provide multiple solutions to address this compatibility issue:

### 1. Quick Fix: Monkey Patching

For a quick fix without modifying any files, add this to the beginning of your script:

```python
import numpy as np

# Add compatibility for deprecated numpy types
np.float = np.float64
np.int = np.int32
```

### 2. Automated Patching (Recommended)

We've created a script that automatically patches your installed `madmom` library to fix all compatibility issues:

```bash
# Navigate to the compatibility fixes directory
cd examples/compatibility_fixes

# Run the fix script
python apply_fixes.py
```

This script will:
1. Find your installed `madmom` library
2. Apply the necessary patches to fix all NumPy compatibility issues
3. Add a monkey patch to the library initialization to ensure compatibility with future releases

### 3. Enhanced Visualization Scripts

We've also provided enhanced scripts that include these fixes along with additional features:

- `examples/compatibility_fixes/test_beatnet_fixed.py`: Basic solution with error handling
- `examples/compatibility_fixes/analyze_test_realtime.py`: Real-time visualization with audio playback
- `examples/compatibility_fixes/enhanced_visualization.py`: Comprehensive visualization with upbeats, rhythmic pattern analysis, and more

To use these scripts:

```bash
cd examples/compatibility_fixes
python enhanced_visualization.py path/to/your/audio.wav
```

## Manual Patch Instructions

If you prefer to manually patch your `madmom` library installation, here are the changes needed:

1. In `madmom/io/__init__.py`:
   - Replace `SEGMENT_DTYPE = [('start', np.float), ('end', np.float), ('label', object)]` with 
     `SEGMENT_DTYPE = [('start', np.float64), ('end', np.float64), ('label', object)]`

2. In `madmom/evaluation/chords.py`:
   - Replace all occurrences of `np.int` with `np.int32`
   - Replace all occurrences of `np.float` with `np.float64`

3. In `madmom/features/beats_hmm.py`:
   - Replace all occurrences of `np.int` with `np.int32`
   - Replace all occurrences of `np.float` with `np.float64`

A patch file is available at `examples/compatibility_fixes/madmom_numpy_fix.patch` if you prefer to use standard patching tools.

## Need Help?

If you encounter any issues with these compatibility fixes, please open an issue on our GitHub repository. 