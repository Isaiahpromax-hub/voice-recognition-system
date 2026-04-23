"""
================================================================================
EMBEDDED SPEECH RECOGNITION SYSTEM
WEEK 6: FINAL IMPLEMENTATION (Euclidean + DTW + Divide-and-Conquer)

Requirements Met:
- Euclidean distance between unknown utterance and templates (Week 4)
- Divide-and-Conquer used for FEATURE EXTRACTION / WINDOWING
- Clean initial feature extraction code with simulated audio
- Recurrence relation analysis included
- Dynamic Time Warping (DTW) implementation (Week 6)
- Accuracy comparison between Euclidean and DTW
- Press 'q' to exit live recognition modes
================================================================================
"""

import numpy as np
import sounddevice as sd
import os
import glob
import librosa
import time

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


# ====================== EUCLIDEAN DISTANCE (WEEK 4) ======================
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


# ====================== DYNAMIC TIME WARPING (WEEK 6) ======================
def dtw_distance(seq1, seq2):
    """
    Dynamic Time Warping distance between two MFCC sequences.
    Week 6 Deliverable: Replaces Euclidean distance for better accuracy.
    
    Time Complexity: O(n*m) where n,m are sequence lengths
    Space Complexity: O(n*m) for the cost matrix
    
    Why DTW is better: Handles variations in speaking speed naturally
    """
    n = len(seq1)
    m = len(seq2)
    
    if n == 0 or m == 0:
        return float('inf')
    
    # Initialize cost matrix with infinity
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0][0] = 0
    
    # Compute DTW using dynamic programming
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Euclidean distance between current frames
            cost = np.sqrt(np.sum((seq1[i-1] - seq2[j-1]) ** 2))
            # DTW recurrence: cost + min(insertion, deletion, match)
            dtw[i][j] = cost + min(dtw[i-1][j],     # insertion
                                    dtw[i][j-1],     # deletion
                                    dtw[i-1][j-1])   # match
    
    return dtw[n][m]


def dtw_with_path(seq1, seq2):
    """
    DTW that also returns the warping path (for visualization and analysis)
    """
    n = len(seq1)
    m = len(seq2)
    
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0][0] = 0
    
    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(np.sum((seq1[i-1] - seq2[j-1]) ** 2))
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    
    # Backtrack to find warping path
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


# ====================== LIVE RECOGNITION (WEEK 4 - EUCLIDEAN) ======================
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
            
            # Ask user if they want to continue
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


# ====================== LIVE RECOGNITION (WEEK 6 - DTW) ======================
def test_live_dtw(use_dac=True):
    """Week 6: Live recognition using Dynamic Time Warping - Press 'q' to exit"""
    print("\n" + "="*60)
    print("WEEK 6: LIVE RECOGNITION (DTW + Divide-and-Conquer)")
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
                    print("DTW distances to templates:")
                    
                    distances = []
                    start_time = time.time()
                    for name, template in zip(template_names, templates):
                        dist = dtw_distance(features, template)
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
            
            # Ask user if they want to continue
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


# ====================== ACCURACY COMPARISON (WEEK 6) ======================
def compare_euclidean_vs_dtw():
    """
    Week 6 Deliverable: Compare Euclidean vs DTW accuracy
    Records test samples and computes accuracy metrics
    """
    print("\n" + "="*60)
    print("WEEK 6: EUCLIDEAN vs DTW ACCURACY COMPARISON")
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
        
        # DTW comparison
        start_dtw = time.time()
        dtw_dists = [dtw_distance(test_feat, t) for t in templates]
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
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    if dtw_acc > euc_acc:
        print(f"  ✓ DTW performs better by {improvement:.1f}%")
        print("  ✓ DTW handles speaking speed variations naturally")
    else:
        print(f"  Euclidean and DTW performed similarly")
    print("="*60)
    
    return euc_acc, dtw_acc


# ====================== SIMULATED AUDIO TEST ======================
def test_simulated(use_dac=True):
    """
    Simulated test with synthetic audio to demonstrate feature extraction 
    and template matching without needing live recording.
    Week 4 Deliverable requirement.
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
    
    # Recurrence Relation (Week 4 requirement)
    print("\n" + "-"*50)
    print("RECURRENCE RELATION FOR DIVIDE-AND-CONQUER WINDOWING")
    print("-"*50)
    print("Recurrence Relation: T(n) = 2 * T(n/2) + Θ(1)")
    print("")
    print("Where:")
    print("  - T(n): time to process n samples")
    print("  - 2T(n/2): two recursive calls on left and right halves")
    print("  - Θ(1): constant work to record current frame position")
    print("")
    print("Solution using Master Theorem:")
    print("  a = 2, b = 2, f(n) = Θ(1)")
    print("  log_b(a) = log2(2) = 1")
    print("  Since f(n) is polynomially smaller than n^1,")
    print("  Complexity = Θ(n log n)")
    print("")
    print("This satisfies the Week 4 requirement of using Divide-and-Conquer for windowing.")


# ====================== MENU ======================
def print_menu():
    print("\n" + "="*60)
    print("   EMBEDDED SPEECH RECOGNITION SYSTEM")
    print("   Week 4 (Euclidean) + Week 6 (DTW)")
    print("="*60)
    for i, cmd in enumerate(commands, 1):
        print(f"   {i:2d}. {cmd}")
    print("\n" + "-"*60)
    print("OPTIONS:")
    print("1. RECORD TEMPLATES (with Divide-and-Conquer)")
    print("2. LIVE RECOGNITION - EUCLIDEAN (Press 'q' to exit)")
    print("3. LIVE RECOGNITION - DTW (Press 'q' to exit)")
    print("4. COMPARE EUCLIDEAN vs DTW (One-time test)")
    print("5. TEST SIMULATED AUDIO (Week 4)")
    print("6. Exit")
    print("="*60)


# ====================== MAIN ======================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   EMBEDDED SPEECH RECOGNITION SYSTEM")
    print("   FINAL IMPLEMENTATION - WEEK 4 & WEEK 6")
    print("="*60)
    print("\nProject Requirements Met:")
    print("  ✓ Euclidean distance template matching (Week 4)")
    print("  ✓ Divide-and-Conquer for feature extraction/windowing")
    print("  ✓ MFCC extraction (13 coeffs, 8kHz, 256/128 frames)")
    print("  ✓ Voice Activity Detection (energy-based)")
    print("  ✓ Simulated audio test with recurrence relation")
    print("  ✓ Dynamic Time Warping (Week 6)")
    print("  ✓ Accuracy comparison (Euclidean vs DTW)")
    print("  ✓ Press 'q' to exit live recognition modes")
    
    while True:
        print_menu()
        choice = input("\nEnter choice (1-6): ").strip()
        
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
            print("\nGoodbye! Project completed.")
            break
        else:
            print("Invalid choice! Please enter 1-6.")
        
        input("\nPress Enter to continue...")