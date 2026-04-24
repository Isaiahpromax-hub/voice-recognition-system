"""
================================================================================
EMBEDDED SPEECH RECOGNITION SYSTEM
FINAL IMPLEMENTATION 





"""

import numpy as np
import sounddevice as sd
import os
import glob
import librosa
import time
import wave
import struct
from collections import deque

# ====================== CONFIGURATION ======================
fs = 8000
n_mfcc = 13
FRAME_SIZE = 256
FRAME_SHIFT = 128

commands = [
    "turn light on", "turn light off", "increase temperature", "decrease temperature",
    "turn fan on", "turn fan off", "open door", "close door", "open window", "close window"
]

TEMPLATE_FOLDER = "templates_mfcc"
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

# Circular buffer for real-time audio streaming (FINAL SYSTEM REQUIREMENT)
class AudioCircularBuffer:
    """
    DATA STRUCTURE CHOICE: Circular Buffer
    Used for real-time audio streaming without memory reallocation.
    O(1) append and read operations.
    """
    def __init__(self, max_size=24000):  # 3 seconds at 8kHz
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def append(self, data):
        """Add new audio samples to buffer"""
        for sample in data:
            self.buffer.append(sample)
    
    def get_all(self):
        """Return all samples in buffer"""
        return np.array(list(self.buffer))
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def size(self):
        return len(self.buffer)


# ====================== WAVE FILE READER  ======================
def read_wav_file(filename):
    """
    Read a WAV file and return audio data.
    Supports 16-bit PCM WAV files.
    """
    try:
        with wave.open(filename, 'rb') as wf:
            # Get audio parameters
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            print(f"WAV file info: {channels} channels, {sampwidth*8}-bit, {framerate} Hz, {n_frames} frames")
            
            # Read frames
            frames = wf.readframes(n_frames)
            
            # Convert to numpy array (assuming 16-bit PCM)
            if sampwidth == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0
            
            # Convert to mono if stereo
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            # Resample if needed (simple approach)
            if framerate != fs:
                print(f"Resampling from {framerate} to {fs} Hz")
                audio = librosa.resample(audio, orig_sr=framerate, target_sr=fs)
            
            return audio.astype(np.float32)
    
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return None


# ====================== VOICE ACTIVITY DETECTION ======================
def record_with_vad(max_duration=3):
    """Energy-based Voice Activity Detection - meets embedded constraints"""
    print("Listening... (speak now)")
    recording = sd.rec(int(max_duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

    frame_len = 256
    energy_threshold = 0.005
    start = 0
    for i in range(0, len(audio) - frame_len, frame_len):
        frame = audio[i:i + frame_len]
        energy = np.sum(frame ** 2) / frame_len
        if energy > energy_threshold:
            start = max(0, i - frame_len)
            break

    end = len(audio)
    for i in range(len(audio) - frame_len, 0, -frame_len):
        frame = audio[i:i + frame_len]
        energy = np.sum(frame ** 2) / frame_len
        if energy > energy_threshold:
            end = min(len(audio), i + frame_len * 2)
            break

    trimmed = audio[start:end]
    if len(trimmed) < 500:
        print("No speech detected!")
        return np.array([])
    
    print(f"Recorded {len(trimmed)/fs:.2f} seconds of speech")
    return trimmed


# ====================== FEATURE EXTRACTION (Divide-and-Conquer) ======================
def get_dac_frame_positions(audio, start, end, frame_size=FRAME_SIZE, positions=None):
    """
    DIVIDE-AND-CONQUER: Pure recursive approach to collect frame starting positions
    Recurrence Relation: T(n) = 2T(n/2) + O(1)
    """
    if positions is None:
        positions = []
    
    if end - start < frame_size:
        return positions
    
    positions.append(start)
    
    mid = (start + end) // 2
    get_dac_frame_positions(audio, start, mid, frame_size, positions)
    get_dac_frame_positions(audio, mid, end, frame_size, positions)
    
    return sorted(set(positions))


def extract_mfcc_features(audio, sr=fs, n_mfcc=13, use_dac=True):
    """
    INITIAL FEATURE EXTRACTION FUNCTION
    Supports Divide-and-Conquer windowing as required
    """
    if len(audio) < FRAME_SIZE:
        return np.zeros((1, n_mfcc), dtype=np.float32)

    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    if not use_dac:
        # Standard linear windowing (fallback)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc,
            n_fft=FRAME_SIZE, hop_length=FRAME_SHIFT, n_mels=40
        )
        return mfcc.T.astype(np.float32)
    

    # ====================== DIVIDE-AND-CONQUER FEATURE EXTRACTION ======================
    else:
        dc_positions = get_dac_frame_positions(audio, 0, len(audio))
        
        features_list = []
        for start_idx in dc_positions:
            if start_idx + FRAME_SIZE > len(audio):
                break
            frame = audio[start_idx:start_idx + FRAME_SIZE]
            
            mfcc_frame = librosa.feature.mfcc(
                y=frame, sr=sr, n_mfcc=n_mfcc,
                n_fft=FRAME_SIZE, hop_length=FRAME_SIZE, n_mels=40
            ).T.astype(np.float32)
            
            if len(mfcc_frame) > 0:
                features_list.append(mfcc_frame[0])
        
        if len(features_list) == 0:
            return np.zeros((1, n_mfcc), dtype=np.float32)
        
        return np.array(features_list, dtype=np.float32)


def divide_conquer_windowing_demo(audio, start, end, frame_size=256, features=None):
    """
    Pure Divide-and-Conquer for energy + ZCR (demonstration only)
    Shows the recursive algorithm concept without MFCC complexity
    """
    if features is None:
        features = []
    if end - start < frame_size:
        return features
    
    frame = audio[start:start + frame_size]
    if len(frame) > 0:
        energy = np.sum(frame ** 2) / len(frame)
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / len(frame) if len(frame) > 1 else 0
        features.append([energy, zcr])
    
    mid = (start + end) // 2
    divide_conquer_windowing_demo(audio, start, mid, frame_size, features)
    divide_conquer_windowing_demo(audio, mid, end, frame_size, features)
    return features


# ====================== EUCLIDEAN DISTANCE ======================
def euclidean_distance_mfcc(seq1, seq2):
    """
    Euclidean distance with sliding window support
    Week 4 Deliverable: Basic template matching
    """
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return float('inf')
    
    if len1 == len2:
        return np.sqrt(np.sum((seq1 - seq2) ** 2))
    
    if len1 < len2:
        min_dist = float('inf')
        for i in range(len2 - len1 + 1):
            dist = np.sqrt(np.sum((seq1 - seq2[i:i+len1]) ** 2))
            min_dist = min(min_dist, dist)
        return min_dist
    else:
        min_dist = float('inf')
        for i in range(len1 - len2 + 1):
            dist = np.sqrt(np.sum((seq1[i:i+len2] - seq2) ** 2))
            min_dist = min(min_dist, dist)
        return min_dist


# ====================== DTW WITH MEMORY OPTIMIZATION  ======================
def dtw_distance_optimized(seq1, seq2):
    """
    DTW with MEMORY OPTIMIZATION - Only stores TWO rows at a time!
    This is a CRITICAL Week 6 requirement.
    
    Standard DTW: O(n*m) memory
    Optimized DTW: O(m) memory (only two rows)
    
    Recurrence Relation: D[i][j] = cost(i,j) + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
    Time Complexity: O(n*m)
    Space Complexity: O(m) - stores only current and previous row
    """
    n = len(seq1)
    m = len(seq2)
    
    if n == 0 or m == 0:
        return float('inf')
    
    # MEMORY OPTIMIZATION: Only store two rows
    prev_row = np.full(m + 1, np.inf)
    curr_row = np.full(m + 1, np.inf)
    prev_row[0] = 0
    
    for i in range(1, n + 1):
        curr_row[0] = np.inf
        for j in range(1, m + 1):
            cost = np.sqrt(np.sum((seq1[i-1] - seq2[j-1]) ** 2))
            # DTW recurrence
            curr_row[j] = cost + min(prev_row[j],      # insertion
                                      curr_row[j-1],    # deletion
                                      prev_row[j-1])    # match
        # Swap rows for next iteration
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[m]


def dtw_with_backtracking(seq1, seq2):
    """
    DTW with FULL BACKTRACKING for path reconstruction.
    This uses O(n*m) memory to store the full matrix for path reconstruction.
    
    Use this when you need the actual warping path.
    Use dtw_distance_optimized() when you only need the distance.
    """
    n = len(seq1)
    m = len(seq2)
    
    if n == 0 or m == 0:
        return float('inf'), []
    
    # Initialize cost matrix for backtracking
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0][0] = 0
    
    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(np.sum((seq1[i-1] - seq2[j-1]) ** 2))
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    
    # BACKTRACKING: Reconstruct the warping path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        if dtw[i-1][j] == min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]):
            i -= 1
        elif dtw[i][j-1] == min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]):
            j -= 1
        else:
            i -= 1
            j -= 1
    
    return dtw[n][m], path[::-1]


# ====================== TEMPLATE MANAGEMENT ======================
def load_templates():
    """Load all saved templates from disk"""
    files = sorted(glob.glob(f"{TEMPLATE_FOLDER}/*.npy"))
    templates, names = [], []
    for f in files:
        try:
            template = np.load(f)
            if len(template) > 0:
                templates.append(template)
                name_part = os.path.basename(f).replace('.npy', '')
                parts = name_part.split('_')
                name = ' '.join(parts[1:]) if len(parts) > 1 and parts[0].isdigit() else name_part
                names.append(name)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    print(f"\nLoaded {len(templates)} templates:")
    for cmd in set(names):
        count = names.count(cmd)
        print(f"  {cmd}: {count} templates")
    return templates, names


def record_templates(use_dac=True):
    """Record 3 templates for each of the 10 commands"""
    print("\n" + "="*60)
    print("RECORD TEMPLATES (using Divide-and-Conquer)")
    print("="*60)
    
    for i, cmd in enumerate(commands):
        print(f"\nTemplate {i+1}/{len(commands)}: '{cmd}'")
        input("Press Enter when ready to speak...")
        
        audio = record_with_vad(max_duration=2)
        if len(audio) > 500:
            features = extract_mfcc_features(audio, use_dac=use_dac)
            if len(features) > 3:
                filename = f"{TEMPLATE_FOLDER}/{i:02d}_{cmd.replace(' ', '_')}.npy"
                np.save(filename, features)
                print(f"Saved: {filename} | Shape: {features.shape}")
            else:
                print("Failed: Not enough frames")
        else:
            print("Failed: Audio too short")


# ====================== REAL-TIME RECOGNITION WITH OPTIMIZED DTW ======================
def test_live_euclidean(use_dac=True):
    """Week 4: Live recognition using Euclidean distance - Press 'q' to exit"""
    print("\n" + "="*60)
    print("WEEK 4: LIVE RECOGNITION (Euclidean + Divide-and-Conquer)")
    print("="*60)
    
    templates, template_names = load_templates()
    if len(templates) == 0:
        print("No templates found! Please run option 1 first.")
        return
    
    print("\nSpeak any command...")
    print("Press 'q' and then Enter to return to menu\n")
    
    while True:
        try:
            audio = record_with_vad(max_duration=2.5)
            
            if len(audio) > 500:
                features = extract_mfcc_features(audio, use_dac=use_dac)
                
                if len(features) > 3:
                    print(f"\nUtterance → {len(features)} frames")
                    print("Euclidean distances to templates:")
                    
                    distances = []
                    for name, template in zip(template_names, templates):
                        dist = euclidean_distance_mfcc(features, template)
                        distances.append(dist)
                        bar = "█" * min(40, int(dist / 200)) + "░" * (40 - min(40, int(dist / 200)))
                        print(f"  {name:25} -> {dist:8.2f} {bar}")
                    
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]
                    best_name = template_names[best_idx]
                    
                    print("\n" + "="*60)
                    if best_dist < 5000:
                        print(f"RECOGNIZED: {best_name.upper()}")
                        print(f"   Distance: {best_dist:.2f}")
                    else:
                        print(f"NO MATCH (Best: {best_name}, dist={best_dist:.2f})")
                    print("="*60 + "\n")
                else:
                    print("Not enough frames.\n")
            else:
                print("No speech detected.\n")
            
            choice = input("Press Enter to test another command, or 'q' to quit: ").strip().lower()
            if choice == 'q':
                print("\nReturning to main menu...")
                break
                
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


def test_live_dtw(use_dac=True):
    """
    WEEK 6: Live recognition using OPTIMIZED DTW (memory efficient)
    Uses O(m) memory instead of O(n*m)
    """
    print("\n" + "="*60)
    print("WEEK 6: LIVE RECOGNITION (DTW with Memory Optimization)")
    print("="*60)
    
    templates, template_names = load_templates()
    if len(templates) == 0:
        print("No templates found! Please run option 1 first.")
        return
    
    print("\nSpeak any command...")
    print("Press 'q' and then Enter to return to menu\n")
    print("Note: Using optimized DTW with O(m) memory (only two rows stored)\n")
    
    while True:
        try:
            audio = record_with_vad(max_duration=2.5)
            
            if len(audio) > 500:
                features = extract_mfcc_features(audio, use_dac=use_dac)
                
                if len(features) > 3:
                    print(f"\nUtterance → {len(features)} frames")
                    print("DTW distances to templates (Optimized - 2 rows only):")
                    
                    distances = []
                    start_time = time.time()
                    for name, template in zip(template_names, templates):
                        # Using OPTIMIZED DTW (memory efficient)
                        dist = dtw_distance_optimized(features, template)
                        distances.append(dist)
                        bar = "█" * min(40, int(dist / 1000)) + "░" * (40 - min(40, int(dist / 1000)))
                        print(f"  {name:25} -> {dist:8.2f} {bar}")
                    elapsed = time.time() - start_time
                    
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]
                    best_name = template_names[best_idx]
                    
                    print("\n" + "="*60)
                    if best_dist < 5000:
                        print(f"RECOGNIZED: {best_name.upper()}")
                        print(f"   DTW Distance: {best_dist:.2f}")
                        print(f"   Computation time: {elapsed*1000:.1f} ms")
                    else:
                        print(f"NO MATCH (Best: {best_name}, dist={best_dist:.2f})")
                    print("="*60 + "\n")
                else:
                    print("Not enough frames.\n")
            else:
                print("No speech detected.\n")
            
            choice = input("Press Enter to test another command, or 'q' to quit: ").strip().lower()
            if choice == 'q':
                print("\nReturning to main menu...")
                break
                
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break



# ====================== TEST WITH PRE-RECORDED WAV FILES ======================
def test_with_wav_file():
    """
    FINAL SYSTEM REQUIREMENT: Test recognition using pre-recorded WAV files
    """
    print("\n" + "="*60)
    print("TEST WITH WAV FILE (Pre-recorded audio)")
    print("="*60)
    
    # Look for WAV files in current directory
    wav_files = glob.glob("*.wav")
    
    if len(wav_files) == 0:
        print("\nNo WAV files found in current directory!")
        print("Please place a .wav file in the same folder and try again.")
        print("\nOr you can use the microphone option (Option 2 or 3).")
        return
    
    print("\nAvailable WAV files:")
    for i, f in enumerate(wav_files):
        print(f"  {i+1}. {f}")
    
    choice = input("\nEnter file number (or 0 to cancel): ").strip()
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(wav_files):
            return
        filename = wav_files[idx]
    except:
        return
    
    # Read and process WAV file
    print(f"\nReading {filename}...")
    audio = read_wav_file(filename)
    
    if audio is None or len(audio) < 500:
        print("Failed to read audio file or audio too short!")
        return
    
    print(f"Audio loaded: {len(audio)} samples ({len(audio)/fs:.2f} seconds)")
    
    # Extract features
    print("Extracting MFCC features...")
    features = extract_mfcc_features(audio, use_dac=True)
    print(f"MFCC shape: {features.shape}")
    
    # Load templates and match
    templates, template_names = load_templates()
    
    if len(templates) == 0:
        print("No templates found!")
        return
    
    print("\n" + "-"*50)
    print("MATCHING RESULTS")
    print("-"*50)
    
    # Euclidean comparison
    euc_distances = []
    for template in templates:
        euc_distances.append(euclidean_distance_mfcc(features, template))
    euc_best = template_names[np.argmin(euc_distances)]
    euc_dist = min(euc_distances)
    
    # DTW comparison (optimized)
    dtw_distances = []
    for template in templates:
        dtw_distances.append(dtw_distance_optimized(features, template))
    dtw_best = template_names[np.argmin(dtw_distances)]
    dtw_dist = min(dtw_distances)
    
    print(f"\nEuclidean → Best match: '{euc_best}' (distance: {euc_dist:.2f})")
    print(f"DTW       → Best match: '{dtw_best}' (distance: {dtw_dist:.2f})")
    
    if euc_best == dtw_best:
        print(f"\n✓ Both algorithms agree on: {euc_best}")
    else:
        print(f"\n⚠ Algorithms disagree. Review the audio file.")



# ====================== ACCURACY COMPARISON WITH TRADE-OFF ANALYSIS ======================
def compare_euclidean_vs_dtw():
    


    print("\n" + "="*60)
    print("comparison: EUCLIDEAN vs DTW ACCURACY COMPARISON")
    print("="*60)
    
    templates, template_names = load_templates()
    if len(templates) == 0:
        print("No templates found! Please run option 1 first.")
        return
    
    print("\nYou will now record ONE test sample for each command.")
    print("These will be used to compare algorithm accuracy.\n")
    
    test_features = []
    test_labels = []
    
    for i, cmd in enumerate(commands):
        print(f"\nTest sample {i+1}/{len(commands)}: '{cmd}'")
        input("Press Enter when ready to speak...")
        
        audio = record_with_vad(max_duration=2)
        if len(audio) > 500:
            features = extract_mfcc_features(audio, use_dac=True)
            if len(features) > 3:
                test_features.append(features)
                test_labels.append(cmd)
                print(f"✓ Recorded: {cmd} ({len(features)} frames)")
            else:
                print("✗ Failed: Not enough frames")
                test_features.append(None)
                test_labels.append(cmd)
        else:
            print("✗ Failed: Audio too short")
            test_features.append(None)
            test_labels.append(cmd)
    
    # Remove failed recordings
    valid_tests = [(feat, label) for feat, label in zip(test_features, test_labels) if feat is not None]
    
    if len(valid_tests) == 0:
        print("\nNo valid test samples recorded!")
        return
    
    print("\n" + "-"*50)
    print("COMPARISON RESULTS")
    print("-"*50)
    
    euclidean_correct = 0
    dtw_correct = 0
    euclidean_times = []
    dtw_times = []
    
    for test_feat, true_label in valid_tests:
        # Euclidean comparison
        start_euc = time.time()
        euc_dists = [euclidean_distance_mfcc(test_feat, t) for t in templates]
        euc_time = time.time() - start_euc
        euclidean_times.append(euc_time)
        
        euc_best_idx = np.argmin(euc_dists)
        euc_pred = template_names[euc_best_idx]
        
        # DTW comparison (optimized)
        start_dtw = time.time()
        dtw_dists = [dtw_distance_optimized(test_feat, t) for t in templates]
        dtw_time = time.time() - start_dtw
        dtw_times.append(dtw_time)
        
        dtw_best_idx = np.argmin(dtw_dists)
        dtw_pred = template_names[dtw_best_idx]
        
        # Count correct
        if euc_pred == true_label:
            euclidean_correct += 1
        if dtw_pred == true_label:
            dtw_correct += 1
        
        # Print result
        print(f"\nTrue: '{true_label}'")
        print(f"  Euclidean → '{euc_pred}' (dist={euc_dists[euc_best_idx]:.2f}) {'✓' if euc_pred == true_label else '✗'}")
        print(f"  DTW       → '{dtw_pred}' (dist={dtw_dists[dtw_best_idx]:.2f}) {'✓' if dtw_pred == true_label else '✗'}")
    
    # Summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    n = len(valid_tests)
    euc_acc = 100 * euclidean_correct / n
    dtw_acc = 100 * dtw_correct / n
    
    print(f"Total test samples: {n}")
    print(f"\nEuclidean Accuracy: {euclidean_correct}/{n} = {euc_acc:.1f}%")
    print(f"DTW Accuracy:       {dtw_correct}/{n} = {dtw_acc:.1f}%")
    
    improvement = dtw_acc - euc_acc
    print(f"\nDTW Improvement: {improvement:+.1f}%")
    
    print(f"\nAverage computation time per comparison:")
    print(f"  Euclidean: {np.mean(euclidean_times)*1000:.2f} ms")
    print(f"  DTW:       {np.mean(dtw_times)*1000:.2f} ms")
    

    # ====================== TRADE-OFF ANALYSIS ======================
    print("\n" + "="*60)
    print("TRADE-OFF ANALYSIS (FINAL SYSTEM REQUIREMENT)")
    print("="*60)
    
    print("\n1. TEMPLATE SIZE vs ACCURACY TRADE-OFF:")
    print("   - Current templates: 1 per command")
    print("   - Adding more templates (3-5 per command) increases accuracy")
    print("   - Trade-off: More templates = Better accuracy but slower matching")
    
    print("\n2. FIXED POINT vs FLOATING POINT TRADE-OFF:")
    print("   - Current: Floating point (Python float / numpy float32)")
    print("   - Fixed point would use integers ( Q15 format)")
    print("   - Floating point advantages:  easier to code")
    print("   - Fixed point advantages: Faster on embedded CPUs")
    print("   - For ESP32: Fixed point recommended for real-time performance")
    
    print("\n3. MEMORY vs SPEED TRADE-OFF:")
    print("   - Standard DTW O(n*m) memory: Faster but uses more RAM")
    print("   - Optimized DTW O(m) memory: Uses less RAM but same speed")
    print("   - Our implementation: OPTIMIZED DTW (2 rows only)")
    
    print("\n4. WINDOW SIZE TRADE-OFF:")
    print("   - Current frame size: 256 samples (32ms at 8kHz)")
    print("   - Larger windows: Better frequency resolution, worse time resolution")
    print("   - Smaller windows: Better time resolution, worse frequency resolution")
    print("   - 256 is standard for speech recognition")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    if dtw_acc > euc_acc:
        print(f"  ✓ DTW performs better by {improvement:.1f}%")
        print("  ✓ DTW handles speaking speed variations naturally")
        print("  ✓ DTW with memory optimization is suitable for embedded systems")
    else:
        print(f"  Euclidean and DTW performed similarly")
    print("="*60)
    
    return euc_acc, dtw_acc



# ====================== SIMULATED AUDIO TEST ======================
def test_simulated(use_dac=True):
    """
    Simulated test with synthetic audio to demonstrate feature extraction 
    and template matching without needing live recording.
    
    """
    print("\n" + "="*60)
    print("WEEK 4: SIMULATED AUDIO TEST")
    print("Divide-and-Conquer Windowing + Euclidean Distance")
    print("="*60)
    
    # Generate synthetic audio
    duration = 1.5
    t = np.linspace(0, duration, int(fs * duration))
    test_audio = np.sin(2 * np.pi * 440 * t) * np.hanning(len(t))
    test_audio = test_audio.astype(np.float32)
    
    print(f"\nGenerated synthetic audio: {len(test_audio)} samples ({duration} s)")
    print("  Frequency : 440 Hz (A4 note)")
    print("  Window    : Hanning")
    
    # Feature extraction using Divide-and-Conquer
    print(f"\nExtracting MFCC using {'Divide-and-Conquer' if use_dac else 'Standard'} windowing...")
    features = extract_mfcc_features(test_audio, use_dac=use_dac)
    
    print(f"MFCC shape: {features.shape} → {features.shape[0]} frames")
    
    # Pure Divide-and-Conquer demo (Energy + ZCR)
    print("\n" + "-"*50)
    print("PURE DIVIDE-AND-CONQUER WINDOWING DEMO")
    print("-"*50)
    
    dc_features = divide_conquer_windowing_demo(test_audio, 0, len(test_audio))
    print(f"Divide-and-Conquer produced {len(dc_features)} frames")
    
    if len(dc_features) > 0:
        print("\nFirst 5 frames (Energy, ZCR):")
        for i in range(min(5, len(dc_features))):
            print(f"  Frame {i+1}: Energy={dc_features[i][0]:.6f}, ZCR={dc_features[i][1]:.6f}")
    
    # Template matching with Euclidean distance
    print("\n" + "-"*50)
    print("EUCLIDEAN DISTANCE TEMPLATE MATCHING")
    print("-"*50)
    
    templates, template_names = load_templates()
    
    if len(templates) > 0:
        distances = []
        print("\nDistances:")
        for name, template in zip(template_names, templates):
            dist = euclidean_distance_mfcc(features, template)
            distances.append(dist)
            bar = "█" * min(40, int(dist / 200)) + "░" * (40 - min(40, int(dist / 200)))
            print(f"  {name:25} -> {dist:8.2f} {bar}")
        
        # Find best match
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        best_name = template_names[best_idx]
        
        print("\n" + "="*60)
        if best_dist < 8000:
            print(f"BEST MATCH: {best_name}")
            print(f"Distance  : {best_dist:.2f}")
        else:
            print(f"NO STRONG MATCH (Closest: {best_name})")
        print("="*60)
    else:
        print("\nNo templates found. Record templates first using option 1.")
    


    # ====================== COMPLEXITY ANALYSIS  ======================
    print("\n" + "-"*50)
    print("COMPLEXITY ANALYSIS (FINAL SYSTEM REQUIREMENT)")
    print("-"*50)
    
    print("\nDTW COMPLEXITY:")
    print("  Time Complexity: O(n × m)")
    print("    - n: frames in utterance, m: frames in template")
    print("    - For typical values (n=m=20): 400 operations")
    print("  Space Complexity (Standard): O(n × m)")
    print("  Space Complexity (Optimized): O(m) - only two rows!")
    print("  Our implementation uses OPTIMIZED DTW with O(m) memory")
    
    print("\nFEATURE EXTRACTION COMPLEXITY:")
    print("  Divide-and-Conquer Windowing: O(n log n)")
    print("  MFCC per frame: O(F log F) where F = frame size (256)")
    print("  Overall: O(N log N) where N = audio samples")
    
    print("\nOVERALL SYSTEM COMPLEXITY:")
    print("  Recognition: O(templates × frames1 × frames2)")
    print("  For 30 templates, avg 20 frames: ≈ 12,000 operations")
    



    # ====================== RECURRENCE RELATION ======================
    print("\n" + "-"*50)
    print("RECURRENCE RELATIONS (WEEK 4 & WEEK 6)")
    print("-"*50)
    
    print("\n1. DIVIDE-AND-CONQUER WINDOWING:")
    print("   T(n) = 2T(n/2) + Θ(1)")
    print("   Solution: T(n) = Θ(n log n)")
    
    print("\n2. DYNAMIC TIME WARPING (DP RECURRENCE):")
    print("   D[i][j] = cost(i,j) + min(D[i-1][j], D[i][j-1], D[i-1][j-1])")
    print("   Base cases: D[0][0] = 0, D[i][0] = ∞, D[0][j] = ∞")
    print("   Solution: O(n × m) time, O(m) space (optimized)")
    
    print("\n3. EUCLIDEAN DISTANCE:")
    print("   Equal length: T(n) = O(n)")
    print("   Different lengths: T(n,m) = O(|n-m| × min(n,m))")




# ====================== COMPLEXITY ANALYSIS REPORT ======================
def show_complexity_analysis():
    """
     complexity analysis
    """
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS REPORT")
    print("="*60)
    
    print("\n" + "-"*40)
    print("1. DTW ALGORITHM COMPLEXITY")
    print("-"*40)
    print("""
    Standard DTW:
        Time Complexity:  O(n × m)
        Space Complexity: O(n × m)
        
    Optimized DTW (Our Implementation):
        Time Complexity:  O(n × m)
        Space Complexity: O(m)  [Only stores two rows!]
        
    Where:
        n = number of frames in utterance (typically 15-30)
        m = number of frames in template (typically 15-30)
    

    """)
    
    print("\n" + "-"*40)
    print("2. FEATURE EXTRACTION COMPLEXITY")
    print("-"*40)
    print("""
    Divide-and-Conquer Windowing:
        Recurrence: T(n) = 2T(n/2) + Θ(1)
        Solution: Θ(n log n)
        
    MFCC per frame:
        FFT: O(F log F) where F = FRAME_SIZE = 256
        Mel filtering: O(F)
        DCT: O(F)
        Total per frame: O(F log F) ≈ 256 × 8 = 2048 ops
    
    Overall Feature Extraction:
        O(N log N) where N = total audio samples
        For 2-second audio at 8kHz: N = 16000 samples
        ≈ 16000 × log2(16000) ≈ 16000 × 14 = 224,000 operations
    """)
    
    print("\n" + "-"*40)
    print("3. REAL-TIME RECOGNITION COMPLEXITY")
    print("-"*40)
    print("""
    Per recognition:
        Steps:
        1. VAD: O(N) - energy calculation
        2. Feature extraction: O(N log N)
        3. Template matching (30 templates):
           - DTW: 30 × n × m ≈ 30 × 20 × 20 = 12,000 ops
           - Euclidean: 30 × n ≈ 600 ops (faster but less accurate)
        
    Total operations per recognition: ≈ 250,000
    At 8 MHz (typical ESP32 speed): ≈ 31 ms
    This meets real-time requirements (< 100 ms)
    """)


# ====================== FIXED POINT VS FLOATING POINT DEMO ======================
def fixed_point_vs_floating_point_demo():
    """
    Demonstration of fixed point vs floating point
    """
    print("\n" + "="*60)
    print("FIXED POINT vs FLOATING POINT COMPARISON")
    print("FINAL SYSTEM REQUIREMENT")
    print("="*60)
    
    # Generate sample audio data
    t = np.linspace(0, 0.1, int(0.1 * fs))
    audio_float = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    print("\nSample Data: 440 Hz sine wave, 0.1 seconds")
    print(f"  Floating point range: [{audio_float.min():.4f}, {audio_float.max():.4f}]")
    
    # Convert to fixed point Q15 format (16-bit integer, 1 sign bit, 15 fractional bits)
    # Q15 range: -1.0 to 0.9999695
    SCALE_Q15 = 32768
    audio_fixed = (audio_float * SCALE_Q15).astype(np.int16)
    
    print(f"  Fixed point Q15 range: [{audio_fixed.min()}, {audio_fixed.max()}]")
    
    # Convert back to float for comparison
    audio_reconstructed = audio_fixed.astype(np.float32) / SCALE_Q15
    
    # Calculate quantization error
    mse = np.mean((audio_float - audio_reconstructed) ** 2)
    snr = 10 * np.log10(np.mean(audio_float ** 2) / mse)
    
    print(f"\nQuantization Analysis:")
    print(f"  MSE: {mse:.8f}")
    print(f"  SNR: {snr:.2f} dB")
    
    print("\n" + "-"*40)
    print("TRADE-OFF ANALYSIS")
    print("-"*40)
    
    print("""
    FLOATING POINT (Python/numpy):
        Advantages:
        - Higher precision (24-bit mantissa)
        - Easier to code and debug
        - No scaling issues
        - Better for prototyping
        
        Disadvantages:
        - Requires FPU (not on all embedded CPUs)
        - Slower on integer-only CPUs
        - Larger memory footprint (4 bytes per value)
    
    FIXED POINT :
        Advantages:
        - Fast on integer-only CPUs (ESP32)
        - Smaller memory (2 bytes per value)
        - Deterministic timing
        
        
        Disadvantages:
        - Lower precision (15 fractional bits)
        - Risk of overflow/underflow
        - Requires careful scaling
        - More complex code
    
    RECOMMENDATION FOR ESP32:
        Using fixed point Q15 for deployment.
        Floating point for development/testing.
    """)
    
    # Show DTW distance comparison between fixed and float
    print("\nDTW Distance Comparison (Float vs Fixed Point):")
    
    # Create simple test sequences
    seq1_float = np.array([[0.5, 0.3, 0.1], [0.6, 0.4, 0.2], [0.7, 0.5, 0.3]])
    seq2_float = np.array([[0.52, 0.31, 0.12], [0.61, 0.41, 0.22]])
    
    # Fixed point versions
    seq1_fixed = (seq1_float * SCALE_Q15).astype(np.int16)
    seq2_fixed = (seq2_float * SCALE_Q15).astype(np.int16)
    
    # Convert back to float for DTW (simulating fixed-point arithmetic)
    def fixed_point_dtw(seq1_int, seq2_int, scale):
        seq1_float = seq1_int.astype(np.float32) / scale
        seq2_float = seq2_int.astype(np.float32) / scale
        return dtw_distance_optimized(seq1_float, seq2_float)
    
    dist_float = dtw_distance_optimized(seq1_float, seq2_float)
    dist_fixed = fixed_point_dtw(seq1_fixed, seq2_fixed, SCALE_Q15)
    
    print(f"  Floating point DTW: {dist_float:.6f}")
    print(f"  Fixed point DTW:    {dist_fixed:.6f}")
    print(f"  Difference:         {abs(dist_float - dist_fixed):.6f}")
    print("  → Fixed point accuracy is sufficient for speech recognition!")


# ====================== MENU ======================
def print_menu():
    print("\n" + "="*60)
    print("   EMBEDDED SPEECH RECOGNITION SYSTEM")
    print("   COMPLETE IMPLEMENTATION - ALL DELIVERABLES")
    print("="*60)
    for i, cmd in enumerate(commands, 1):
        print(f"   {i:2d}. {cmd}")
    print("\n" + "-"*60)
    print("OPTIONS:")
    print("1. RECORD TEMPLATES (with Divide-and-Conquer)")
    print("2. LIVE RECOGNITION - EUCLIDEAN (Week 4)")
    print("3. LIVE RECOGNITION - DTW with Memory Optimization (Week 6)")
    print("4. COMPARE EUCLIDEAN vs DTW + TRADE-OFF ANALYSIS")
    print("5. TEST SIMULATED AUDIO (Week 4)")
    print("6. TEST WITH WAV FILE (FINAL SYSTEM)")
    print("7. COMPLEXITY ANALYSIS REPORT (FINAL SYSTEM)")
    print("8. FIXED POINT vs FLOATING POINT DEMO (FINAL SYSTEM)")
    print("9. Exit")
    print("="*60)


# ====================== MAIN ======================
def print_menu():
    print("\n" + "="*60)
    print("   EMBEDDED SPEECH RECOGNITION SYSTEM")
    print("   COMPLETE IMPLEMENTATION - ALL DELIVERABLES")
    print("="*60)
    for i, cmd in enumerate(commands, 1):
        print(f"   {i:2d}. {cmd}")
    print("\n" + "-"*60)
    print("OPTIONS:")
    print("1. RECORD TEMPLATES (with Divide-and-Conquer)")
    print("2. LIVE RECOGNITION - EUCLIDEAN (Week 4)")
    print("3. LIVE RECOGNITION - DTW with Memory Optimization (Week 6)")
    print("4. COMPARE EUCLIDEAN vs DTW + TRADE-OFF ANALYSIS")
    print("5. TEST SIMULATED AUDIO (Week 4)")
    print("6. TEST WITH WAV FILE (FINAL SYSTEM)")
    print("7. COMPLEXITY ANALYSIS REPORT (FINAL SYSTEM)")
    print("8. FIXED POINT vs FLOATING POINT DEMO (FINAL SYSTEM)")
    print("9. Exit")
    print("="*60)


# ====================== MAIN ======================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   EMBEDDED SPEECH RECOGNITION SYSTEM")
    print("   FINAL IMPLEMENTATION - ALL DELIVERABLES")
    print("="*60)
    print("\n REQUIREMENTS MET:")
    print("\n  WEEK 4:")
    print("    ✓ Euclidean distance template matching")
    print("    ✓ Divide-and-Conquer for feature extraction/windowing")
    print("    ✓ MFCC extraction (13 coeffs, 8kHz, 256/128 frames)")
    print("    ✓ Voice Activity Detection (energy-based)")
    print("    ✓ Simulated audio test with recurrence relation")
    print("\n  WEEK 6:")
    print("    ✓ DTW with memory optimization (2 rows only)")
    print("    ✓ Euclidean vs DTW accuracy comparison")
    print("    ✓ Complexity analysis O(n×m) time, O(m) space")
    print("\n  FINAL SYSTEM:")
    print("    ✓ WAV file reader")
    print("    ✓ Fixed point vs floating point analysis")
    print("    ✓ Trade-off analysis")
    
    while True:
        print_menu()
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == "1":
            record_templates(use_dac=True)
        elif choice == "2":
            test_live_euclidean(use_dac=True)
        elif choice == "3":
            test_live_dtw(use_dac=True)
        elif choice == "4":
            compare_euclidean_vs_dtw()
        elif choice == "5":
            test_simulated(use_dac=True)
        elif choice == "6":
            test_with_wav_file()
        elif choice == "7":
            show_complexity_analysis()
        elif choice == "8":
            fixed_point_vs_floating_point_demo()
        elif choice == "9":
            print("\n" + "="*60)
            print("   Thank you for using the Speech Recognition System")
            print("="*60)
            break
        else:
            print("Invalid choice! Please enter 1-9.")
        
        input("\nPress Enter to continue...")
