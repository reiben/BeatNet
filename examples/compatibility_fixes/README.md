# BeatNet Compatibility Fixes for NumPy 1.20+

This directory contains fixes and enhancements for using BeatNet with modern NumPy versions (1.20+).

## Problem

The original BeatNet library and its dependency `madmom` use deprecated NumPy types like `np.float` and `np.int` which were removed in NumPy 1.20. This causes errors when running BeatNet with recent versions of NumPy.

## Solution

We've created several scripts that address these compatibility issues:

1. `test_beatnet_fixed.py` - Basic solution with monkey patching and error handling
2. `analyze_test_realtime.py` - Real-time visualization with audio playback
3. `enhanced_visualization.py` - Comprehensive visualization with upbeats, beat strength, and rhythmic pattern analysis

## Usage

### Basic Usage

```bash
python test_beatnet_fixed.py
```

### Real-time Visualization

```bash
python analyze_test_realtime.py
```

### Enhanced Visualization

```bash
python enhanced_visualization.py
```

You can also specify an audio file as an argument to any of these scripts:

```bash
python enhanced_visualization.py path/to/your/audio.wav
```

## Features

- **Compatibility fixes** for NumPy 1.20+ via monkey patching
- **Meter (time signature) detection**
- **Tempo estimation** in BPM
- **Upbeat detection**
- **Beat strength visualization**
- **Rhythmic pattern analysis**
- **Real-time visualization** with audio playback
- **Interactive controls** for navigating through the audio

## Implementation Details

The scripts use the following approach:

1. Add compatibility for deprecated NumPy types:
   ```python
   np.float = np.float64
   np.int = np.int32
   ```

2. Fix array shape issues in madmom's downbeats module by monkey patching
3. Add error handling to ensure robustness

## Requirements

- BeatNet
- NumPy (1.20+)
- Matplotlib
- PyGame (for audio playback)
- SciPy

## Output

The scripts generate:
- A text file with detailed beat analysis
- Visual representation of beats, downbeats, and upbeats
- (When using enhanced visualization) Additional insights into the rhythmic structure 