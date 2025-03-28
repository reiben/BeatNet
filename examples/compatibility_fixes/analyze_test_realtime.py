import numpy as np
import warnings
import sys
import os
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import pygame
from scipy.io import wavfile

# Suppress the deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add compatibility for deprecated numpy types
np.float = np.float64
np.int = np.int32

# Monkey patch the downbeats module to fix the array shape issue
def apply_patches():
    import madmom.features.downbeats
    
    # Original function
    original_process = madmom.features.downbeats.DBNDownBeatTrackingProcessor.process
    
    # Patched function
    def patched_process(self, activations, **kwargs):
        # pylint: disable=arguments-differ
        import itertools as it
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))
        # (parallel) decoding of the activations with HMM
        results = list(self.map(madmom.features.downbeats._process_dbn, 
                                zip(self.hmms, it.repeat(activations))))
        
        # FIX: Extract log probabilities separately to avoid array shape issues
        log_probs = [result[1] for result in results]
        best = np.argmax(log_probs)
        
        # the best path through the state space
        path, _ = results[best]
        # the state space and observation model of the best HMM
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        # the positions inside the pattern (0..num_beats)
        positions = st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=np.int32)  # Use np.int32 explicitly
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are >= 1
            beat_range = om.pointers[path] >= 1
            # get all change points between True and False (cast to int before)
            idx = np.nonzero(np.diff(beat_range.astype(np.int32)))[0] + 1  # Use np.int32 explicitly
            # if the first frame is in the beat range, add a change at frame 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    # pick the frame with the highest activations value
                    # Note: we look for both beats and down-beat activations;
                    #       since np.argmax works on the flattened array, we
                    #       need to divide by 2
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            # transitions are the points where the beat numbers change
            # FIXME: we might miss the first or last beat!
            #        we could calculate the interval towards the beginning/end
            #        to decide whether to include these points
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        # return the beat positions (converted to seconds) and beat numbers
        return np.vstack(((beats + first) / float(self.fps),
                          beat_numbers[beats])).T
    
    # Apply the monkey patch
    madmom.features.downbeats.DBNDownBeatTrackingProcessor.process = patched_process

# Apply all patches
apply_patches()

def detect_meter(output):
    """
    Analyze the beat pattern to determine the meter (time signature).
    """
    # Extract downbeats
    downbeats = [beat[0] for beat in output if beat[1] == 1]
    
    # Need at least 2 downbeats to calculate intervals
    if len(downbeats) < 3:
        return "Unknown"
    
    # Calculate intervals between downbeats
    intervals = []
    for i in range(1, len(downbeats)):
        intervals.append(downbeats[i] - downbeats[i-1])
    
    # Calculate average interval
    avg_interval = sum(intervals) / len(intervals)
    
    # Count beats between downbeats to determine meter
    beat_counts = []
    downbeat_indices = [i for i, beat in enumerate(output) if beat[1] == 1]
    
    for i in range(1, len(downbeat_indices)):
        beat_count = downbeat_indices[i] - downbeat_indices[i-1]
        beat_counts.append(beat_count)
    
    # Find the most common beat count
    if not beat_counts:
        return "Unknown"
    
    # Count occurrences of each beat count
    from collections import Counter
    counter = Counter(beat_counts)
    most_common = counter.most_common(1)[0][0]
    
    # Determine time signature
    if most_common == 2:
        return "2/4"
    elif most_common == 3:
        return "3/4"
    elif most_common == 4:
        return "4/4"
    elif most_common == 6:
        return "6/8"
    else:
        return f"{most_common}/4"

class AudioPlayer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.playing = False
        self.current_time = 0
        
        try:
            # Try to read with scipy
            self.sample_rate, self.audio_data = wavfile.read(audio_file)
            
            # Convert stereo to mono if needed
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
                
            # Initialize pygame for audio playback
            pygame.mixer.init(frequency=self.sample_rate)
            pygame.mixer.music.load(audio_file)
            self.initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize audio player: {str(e)}")
            print("Visualization will work but audio playback will be disabled.")
            self.initialized = False
    
    def play(self):
        """Start audio playback"""
        if not self.initialized:
            print("Audio playback is not available")
            return
            
        pygame.mixer.music.play()
        self.playing = True
        self.start_time = time.time()
    
    def stop(self):
        """Stop audio playback"""
        if not self.initialized:
            return
            
        pygame.mixer.music.stop()
        self.playing = False
    
    def get_current_position(self):
        """Get current playback position in seconds"""
        if self.playing and self.initialized:
            self.current_time = time.time() - self.start_time
        return self.current_time

class RealtimeVisualizer:
    def __init__(self, audio_file, beat_data, duration=60):
        self.audio_player = AudioPlayer(audio_file)
        self.beat_data = beat_data
        
        # Handle case when duration is None
        if duration is None:
            duration = 60
            
        self.duration = min(duration, beat_data[-1][0] if len(beat_data) > 0 else 60)
        
        # Separate beats and downbeats
        self.beats = [b[0] for b in beat_data if b[1] == 0]
        self.downbeats = [b[0] for b in beat_data if b[1] == 1]
        
        # Detect meter
        self.meter = detect_meter(beat_data)
        
        # Setup the figure and animation
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.current_time_line = Line2D([0, 0], [0, 1], color='r', linewidth=2)
        self.ax.add_line(self.current_time_line)
        
        # Plot all beat markers
        for beat in self.beats:
            self.ax.axvline(x=beat, color='b', linestyle='-', alpha=0.3)
        
        for downbeat in self.downbeats:
            self.ax.axvline(x=downbeat, color='r', linestyle='-', alpha=0.5, linewidth=2)
        
        # Add meter information
        self.ax.set_title(f'Beat and Downbeat Detection - Meter: {self.meter}')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_yticks([])
        self.ax.set_xlim(0, self.duration)
        
        # Add a slider to control the visible window
        from matplotlib.widgets import Slider
        self.slider_ax = plt.axes([0.2, 0.01, 0.65, 0.03])
        self.slider = Slider(self.slider_ax, 'Position', 0, max(self.duration - 10, 1), valinit=0)
        self.slider.on_changed(self.update_window)
        
        # Start the animation
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=50, blit=True)
        
        # Play button
        from matplotlib.widgets import Button
        self.button_ax = plt.axes([0.05, 0.01, 0.1, 0.04])
        self.button = Button(self.button_ax, 'Play/Pause')
        self.button.on_clicked(self.toggle_playback)
    
    def update(self, frame):
        """Update the visualization for each animation frame"""
        current_pos = self.audio_player.get_current_position()
        self.current_time_line.set_xdata([current_pos, current_pos])
        
        # Update the slider if playing
        if self.audio_player.playing and current_pos > self.slider.val + 9:
            self.slider.set_val(current_pos - 5)
        
        return [self.current_time_line]
    
    def update_window(self, val):
        """Update the visible window based on slider position"""
        self.ax.set_xlim(val, val + 10)
        self.fig.canvas.draw_idle()
    
    def toggle_playback(self, event):
        """Toggle between play and pause"""
        if not self.audio_player.initialized:
            print("Audio playback is not available")
            return
        
        if self.audio_player.playing:
            self.audio_player.stop()
        else:
            # Set position based on current view
            pos_seconds = self.slider.val
            try:
                pygame.mixer.music.play(start=pos_seconds)
                self.audio_player.playing = True
                self.audio_player.start_time = time.time() - pos_seconds
            except Exception as e:
                print(f"Error starting playback: {e}")
                self.audio_player.playing = False
    
    def show(self):
        """Show the visualization"""
        plt.show()

def analyze_file(audio_file, max_duration=None):
    """Analyze audio file and visualize beats in real-time"""
    try:
        from BeatNet.BeatNet import BeatNet
        
        # Initialize BeatNet in offline mode
        print("Initializing BeatNet...")
        model_num = 1  # Model choice (1-3)
        mode = 'offline'  # Use offline mode for analysis
        inference_model = 'DBN'  # Use DBN for non-causal inference
        
        estimator = BeatNet(model_num, mode=mode, inference_model=inference_model, plot=[], thread=False)
        
        if os.path.exists(audio_file):
            print(f"Processing file: {audio_file}...")
            start_time = time.time()
            output = estimator.process(audio_file)
            end_time = time.time()
            
            # Print results
            print(f"Processing completed in {end_time - start_time:.2f} seconds.")
            print(f"Found {len(output)} beats/downbeats.")
            
            # Detect and display meter
            meter = detect_meter(output)
            print(f"Detected meter (time signature): {meter}")
            
            # First few beats
            print("First 10 beats/downbeats (time in seconds, beat/downbeat):")
            for i in range(min(10, len(output))):
                print(f"{output[i][0]:.2f}s - {'Downbeat' if output[i][1] == 1 else 'Beat'}")
            
            # Save results to a text file
            with open("TEST_beats.txt", "w") as f:
                f.write(f"Beat analysis for {audio_file}\n")
                f.write(f"Total beats/downbeats: {len(output)}\n")
                f.write(f"Detected meter (time signature): {meter}\n")
                f.write("\nTime (s) - Type\n")
                for beat in output:
                    f.write(f"{beat[0]:.4f} - {'Downbeat' if beat[1] == 1 else 'Beat'}\n")
                f.write(f"\nAnalysis completed in {end_time - start_time:.2f} seconds.\n")
            
            print(f"Beat analysis saved to TEST_beats.txt")
            
            # Create real-time visualization
            max_dur = max_duration if max_duration else None
            visualizer = RealtimeVisualizer(audio_file, output, max_dur)
            visualizer.show()
            
        else:
            print(f"Error: Audio file {audio_file} not found")
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    
    # Default file to analyze
    audio_file = "TEST.wav"
    
    # Check command line arguments for audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    # Analyze the file with real-time visualization
    analyze_file(audio_file)
    
    # Clean up pygame
    pygame.quit() 