import numpy as np
import warnings
import sys
import os
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import pygame
from scipy.io import wavfile
from collections import defaultdict

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

def calculate_tempo(beat_times):
    """
    Calculate tempo in BPM from beat times.
    """
    if len(beat_times) < 2:
        return 0
        
    # Calculate intervals between beats
    intervals = []
    for i in range(1, len(beat_times)):
        intervals.append(beat_times[i] - beat_times[i-1])
    
    # Calculate average interval
    if not intervals:
        return 0
        
    avg_interval = sum(intervals) / len(intervals)
    
    # Convert to BPM
    if avg_interval > 0:
        return 60.0 / avg_interval
    return 0

def identify_upbeats(output, meter):
    """
    Identify upbeats based on meter and beat positions.
    """
    # Get beats per measure based on meter
    if meter == "2/4":
        beats_per_measure = 2
    elif meter == "3/4":
        beats_per_measure = 3  
    elif meter == "4/4":
        beats_per_measure = 4
    elif meter == "6/8":
        beats_per_measure = 6
    elif "/" in meter:
        beats_per_measure = int(meter.split("/")[0])
    else:
        beats_per_measure = 4  # default
    
    upbeats = []
    
    # Group beats by measure
    downbeat_indices = [i for i, beat in enumerate(output) if beat[1] == 1]
    
    # Last beat before a downbeat is typically an upbeat
    for i in range(len(downbeat_indices)):
        if i > 0:  # Not the first downbeat
            prev_downbeat_idx = downbeat_indices[i-1]
            curr_downbeat_idx = downbeat_indices[i]
            
            # If there are beats between downbeats
            if curr_downbeat_idx - prev_downbeat_idx > 1:
                # Last beat before current downbeat is an upbeat
                upbeat_idx = curr_downbeat_idx - 1
                upbeats.append(output[upbeat_idx][0])
    
    return upbeats

def analyze_rhythmic_pattern(output, meter):
    """
    Analyze the rhythmic pattern based on beat positions and meter.
    """
    # Get beats per measure from meter
    if meter == "2/4":
        beats_per_measure = 2
    elif meter == "3/4":
        beats_per_measure = 3  
    elif meter == "4/4":
        beats_per_measure = 4
    elif meter == "6/8":
        beats_per_measure = 6
    elif "/" in meter:
        beats_per_measure = int(meter.split("/")[0])
    else:
        beats_per_measure = 4  # default

    # Group beats by measure
    measures = []
    current_measure = []
    beat_count = 0
    
    for beat in output:
        # If it's a downbeat and not the first beat, start a new measure
        if beat[1] == 1 and beat_count > 0:
            measures.append(current_measure)
            current_measure = [beat]
        else:
            current_measure.append(beat)
        beat_count += 1
    
    # Add the last measure
    if current_measure:
        measures.append(current_measure)
    
    # Analyze beat distribution within measures
    if len(measures) < 3:
        return "Insufficient data for pattern analysis"
    
    # Calculate average number of beats per measure
    avg_beats_per_measure = sum(len(m) for m in measures) / len(measures)
    
    # Calculate average time intervals within measures
    interval_patterns = defaultdict(int)
    
    for measure in measures:
        if len(measure) < 2:
            continue
        
        # Calculate intervals between beats within the measure
        intervals = []
        for i in range(1, len(measure)):
            intervals.append(round((measure[i][0] - measure[i-1][0]) * 100) / 100)  # Round to 2 decimal places
        
        # Convert intervals to string representation
        pattern = "-".join([f"{i:.2f}" for i in intervals])
        interval_patterns[pattern] += 1
    
    # Find the most common pattern
    if not interval_patterns:
        return "No consistent pattern detected"
    
    most_common_pattern = max(interval_patterns.items(), key=lambda x: x[1])
    
    return most_common_pattern[0]

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

class EnhancedVisualizer:
    def __init__(self, audio_file, beat_data, duration=60):
        self.audio_player = AudioPlayer(audio_file)
        self.beat_data = beat_data
        
        # Handle case when duration is None
        if duration is None:
            duration = 60
            
        self.duration = min(duration, beat_data[-1][0] if len(beat_data) > 0 else 60)
        
        # Separate beats by type
        self.beats = [b[0] for b in beat_data if b[1] == 0]
        self.downbeats = [b[0] for b in beat_data if b[1] == 1]
        
        # Extract additional musical information
        self.meter = detect_meter(beat_data)
        self.tempo = calculate_tempo(sorted(self.beats + self.downbeats))
        self.upbeats = identify_upbeats(beat_data, self.meter)
        self.rhythmic_pattern = analyze_rhythmic_pattern(beat_data, self.meter)
        
        # Setup the figure and axes
        self.fig = plt.figure(figsize=(15, 8))
        self.grid = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # Main beat visualization
        self.ax_main = self.fig.add_subplot(self.grid[0])
        self.current_time_line = Line2D([0, 0], [0, 1], color='black', linewidth=2)
        self.ax_main.add_line(self.current_time_line)
        
        # Add beat markers with different colors and styles
        for beat in self.beats:
            self.ax_main.axvline(x=beat, color='blue', linestyle='-', alpha=0.3, linewidth=1)
        
        for downbeat in self.downbeats:
            self.ax_main.axvline(x=downbeat, color='red', linestyle='-', alpha=0.6, linewidth=2)
            
        for upbeat in self.upbeats:
            self.ax_main.axvline(x=upbeat, color='green', linestyle='-', alpha=0.5, linewidth=1.5)
        
        # Add meter and tempo information to title
        self.ax_main.set_title(f'Beat Detection - Meter: {self.meter}, Tempo: {self.tempo:.1f} BPM', fontsize=14)
        self.ax_main.set_xlabel('Time (s)', fontsize=12)
        self.ax_main.set_yticks([])
        self.ax_main.set_xlim(0, min(self.duration, 10))
        
        # Create legend
        self.ax_main.legend(
            [Line2D([0], [0], color='red', linewidth=2), 
             Line2D([0], [0], color='blue', linewidth=1),
             Line2D([0], [0], color='green', linewidth=1.5)],
            ['Downbeat', 'Beat', 'Upbeat'],
            loc='upper right'
        )
        
        # Add measure markers (alternating background colors)
        if len(self.downbeats) >= 2:
            for i in range(len(self.downbeats) - 1):
                if i % 2 == 0:
                    rect = patches.Rectangle(
                        (self.downbeats[i], 0), 
                        self.downbeats[i+1] - self.downbeats[i], 
                        1, 
                        alpha=0.1, 
                        facecolor='lightblue',
                        transform=self.ax_main.get_xaxis_transform()
                    )
                    self.ax_main.add_patch(rect)
        
        # Second subplot for beat strength visualization
        self.ax_strength = self.fig.add_subplot(self.grid[1], sharex=self.ax_main)
        self.ax_strength.set_title('Beat Strength', fontsize=12)
        
        # Create beat strength visualization based on beat position in measure
        strength_x = []
        strength_y = []
        
        beat_in_measure = 1
        prev_time = 0
        
        for i, beat in enumerate(beat_data):
            beat_time = beat[0]
            is_downbeat = beat[1] == 1
            
            if is_downbeat:
                beat_in_measure = 1
            else:
                beat_in_measure += 1
            
            # Beat strength decreases as we move further from the downbeat
            if is_downbeat:
                strength = 1.0  # Maximum strength for downbeat
            elif beat_time in self.upbeats:
                strength = 0.8  # Upbeats are strong
            else:
                # Normalize within the measure based on position
                beats_in_meter = int(self.meter.split('/')[0]) if '/' in self.meter else 4
                strength = max(0.3, 1.0 - (beat_in_measure - 1) / beats_in_meter)
            
            strength_x.append(beat_time)
            strength_y.append(strength)
            
            prev_time = beat_time
        
        self.ax_strength.bar(strength_x, strength_y, width=0.1, color='purple', alpha=0.7)
        self.ax_strength.set_ylim(0, 1.1)
        self.ax_strength.set_ylabel('Strength')
        
        # Third subplot for rhythmic pattern visualization
        self.ax_pattern = self.fig.add_subplot(self.grid[2], sharex=self.ax_main)
        self.ax_pattern.set_title(f'Rhythmic Pattern: {self.rhythmic_pattern}', fontsize=12)
        self.ax_pattern.set_yticks([])
        
        # Create measure rectangles
        if len(self.downbeats) >= 2:
            for i in range(len(self.downbeats) - 1):
                start = self.downbeats[i]
                end = self.downbeats[i+1]
                
                # Get all beats in this measure
                measure_beats = [b for b in sorted(self.beats + self.downbeats) if start <= b < end]
                
                # Visualize the beat pattern within the measure
                for j, beat_time in enumerate(measure_beats):
                    color = 'red' if beat_time in self.downbeats else 'blue'
                    marker_size = 15 if beat_time in self.downbeats else 10
                    alpha = 0.8 if beat_time in self.downbeats else 0.6
                    self.ax_pattern.scatter(beat_time, 0.5, color=color, s=marker_size, alpha=alpha)
                    
                    # Connect beats with lines
                    if j < len(measure_beats) - 1:
                        self.ax_pattern.plot(
                            [measure_beats[j], measure_beats[j+1]], [0.5, 0.5], 
                            'k-', alpha=0.3
                        )
        
        # Fourth subplot for additional information
        self.ax_info = self.fig.add_subplot(self.grid[3])
        self.ax_info.axis('off')  # Hide axes
        
        # Display additional information as text
        info_text = (
            f"Audio File: {os.path.basename(audio_file)}\n"
            f"Total Beats: {len(self.beats)}\n"
            f"Total Downbeats: {len(self.downbeats)}\n"
            f"Total Upbeats: {len(self.upbeats)}\n"
            f"Meter: {self.meter}\n"
            f"Tempo: {self.tempo:.1f} BPM\n"
            f"Beat Pattern: {self.rhythmic_pattern}"
        )
        
        self.ax_info.text(0.01, 0.5, info_text, fontsize=12, verticalalignment='center')
        
        # Add a slider to control the visible window
        from matplotlib.widgets import Slider
        self.slider_ax = plt.axes([0.2, 0.01, 0.65, 0.02])
        self.slider = Slider(self.slider_ax, 'Position', 0, max(self.duration - 10, 1), valinit=0)
        self.slider.on_changed(self.update_window)
        
        # Play button
        from matplotlib.widgets import Button
        self.button_ax = plt.axes([0.05, 0.01, 0.1, 0.02])
        self.button = Button(self.button_ax, 'Play/Pause')
        self.button.on_clicked(self.toggle_playback)
        
        plt.tight_layout()
        
        # Start the animation
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=50, blit=True)
    
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
        window_size = 10  # Show 10 seconds at a time
        self.ax_main.set_xlim(val, val + window_size)
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
    """Analyze audio file and create enhanced visualization"""
    try:
        from BeatNet.BeatNet import BeatNet
        
        # Initialize BeatNet
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
            
            # Detect meter and tempo
            meter = detect_meter(output)
            all_beats = sorted([beat[0] for beat in output])
            tempo = calculate_tempo(all_beats)
            upbeats = identify_upbeats(output, meter)
            
            print(f"Detected meter: {meter}")
            print(f"Estimated tempo: {tempo:.1f} BPM")
            print(f"Identified {len(upbeats)} upbeats")
            
            # Save results to a text file
            with open(f"{os.path.splitext(audio_file)[0]}_analysis.txt", "w") as f:
                f.write(f"Enhanced Beat Analysis for {audio_file}\n")
                f.write(f"Total beats/downbeats: {len(output)}\n")
                f.write(f"Detected meter: {meter}\n")
                f.write(f"Estimated tempo: {tempo:.1f} BPM\n")
                f.write(f"Number of upbeats: {len(upbeats)}\n")
                f.write(f"Rhythmic pattern: {analyze_rhythmic_pattern(output, meter)}\n\n")
                
                f.write("Time (s) - Type\n")
                for beat in output:
                    beat_type = "Downbeat" if beat[1] == 1 else "Beat"
                    # Check if it's an upbeat
                    if beat[0] in upbeats:
                        beat_type = "Upbeat"
                        
                    f.write(f"{beat[0]:.4f} - {beat_type}\n")
                f.write(f"\nAnalysis completed in {end_time - start_time:.2f} seconds.\n")
            
            print(f"Enhanced beat analysis saved to {os.path.splitext(audio_file)[0]}_analysis.txt")
            
            # Create enhanced visualization
            max_dur = max_duration if max_duration else None
            visualizer = EnhancedVisualizer(audio_file, output, max_dur)
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
    
    # Analyze the file with enhanced visualization
    analyze_file(audio_file)
    
    # Clean up pygame
    pygame.quit() 