# BeatNet Examples

This directory contains examples and additional utilities for using the BeatNet library.

## Contents

- **compatibility_fixes/** - Contains scripts and utilities to ensure compatibility with NumPy 1.20+. See [NumPy_Compatibility.md](/NumPy_Compatibility.md) in the root directory for details.
  - `test_beatnet_fixed.py` - Basic solution with monkey patching and error handling
  - `analyze_test_realtime.py` - Real-time visualization with audio playback
  - `enhanced_visualization.py` - Comprehensive visualization with upbeats, beat strength, and rhythmic pattern analysis
  - `apply_fixes.py` - Script to automatically patch your installed madmom library
  - `madmom_numpy_fix.patch` - Patch file for manual application

## Usage

See the README.md within each subdirectory for specific usage instructions. 