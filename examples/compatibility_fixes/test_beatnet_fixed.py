import numpy as np
import warnings

# Suppress the deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add compatibility for deprecated numpy types
np.float = np.float64
np.int = np.int32

import matplotlib.pyplot as plt
import time
import os

try:
    from BeatNet.BeatNet import BeatNet
    
    # Initialize BeatNet in offline mode
    print("Initializing BeatNet...")
    model_num = 1  # Model choice (1-3)
    mode = 'offline'  # Use offline mode since we're just testing
    inference_model = 'DBN'  # Use DBN for non-causal inference
    
    try:
        estimator = BeatNet(model_num, mode=mode, inference_model=inference_model, plot=[], thread=False)
        
        # Process audio file
        audio_file = "test_audio.mp3"
        if os.path.exists(audio_file):
            print(f"Processing file: {audio_file}...")
            start_time = time.time()
            output = estimator.process(audio_file)
            end_time = time.time()
            
            # Print results
            print(f"Processing completed in {end_time - start_time:.2f} seconds.")
            print(f"Found {len(output)} beats/downbeats.")
            print("First 10 beats/downbeats (time in seconds, beat/downbeat):")
            for i in range(min(10, len(output))):
                print(f"{output[i][0]:.2f}s - {'Downbeat' if output[i][1] == 1 else 'Beat'}")
            
            # Plot the beats and downbeats
            if len(output) > 0:
                plt.figure(figsize=(12, 4))
                
                # Separate beats and downbeats
                beats = [b[0] for b in output if b[1] == 0]
                downbeats = [b[0] for b in output if b[1] == 1]
                
                # Plot beats as vertical lines
                for beat in beats:
                    plt.axvline(x=beat, color='b', linestyle='-', alpha=0.5)
                
                # Plot downbeats as vertical lines (thicker)
                for downbeat in downbeats:
                    plt.axvline(x=downbeat, color='r', linestyle='-', linewidth=2)
                
                plt.xlim(0, max(output[-1][0], 10))  # Set x-axis limits
                plt.title('Beat and Downbeat Detection')
                plt.xlabel('Time (s)')
                plt.yticks([])  # Hide y-axis ticks
                plt.savefig('beat_detection_results.png')
                print("Plot saved as 'beat_detection_results.png'")
        else:
            print(f"Error: Audio file {audio_file} not found")
    except Exception as e:
        print(f"Error initializing or running BeatNet: {e}")
        import traceback
        traceback.print_exc()
except ImportError as e:
    print(f"Error importing BeatNet: {e}")
    import traceback
    traceback.print_exc() 