"""
================================================================================
EMBEDDED SPEECH RECOGNITION SYSTEM
WEEK 4: TEMPLATE MATCHING FOUNDATION 

 Week 4 Deliverables:
- Euclidean distance between unknown utterance and templates
- Divide-and-Conquer used for FEATURE EXTRACTION / WINDOWING
- Clean initial feature extraction code
- Recurrence relation analysis included
================================================================================
"""

import numpy as np
import sounddevice as sd
import os
import glob
import librosa


try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice: pip install sounddevice")
    exit(1)


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


# ====================== FEATURE EXTRACTION (Initial Code + DAC) ======================
def extract_mfcc_features(audio, sr=fs, n_mfcc=13, use_dac=False):
    """
    INITIAL FEATURE EXTRACTION FUNCTION (Week 4 requirement)
    Supports both standard and Divide-and-Conquer windowing.
    """
    if len(audio) < FRAME_SIZE:
        return np.zeros((1, n_mfcc), dtype=np.float32)

    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    if not use_dac:
        # Standard linear windowing using librosa
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


def get_dac_frame_positions(audio, start, end, frame_size=FRAME_SIZE, positions=None):
    """ Pure Divide-and-Conquer to collect frame starting positions """
    if positions is None:
        positions = []
    
    if end - start < frame_size:
        return positions
    
    positions.append(start)
    
    mid = (start + end) // 2
    get_dac_frame_positions(audio, start, mid, frame_size, positions)
    get_dac_frame_positions(audio, mid, end, frame_size, positions)
    
    return sorted(set(positions))


# ====================== EUCLIDEAN DISTANCE ======================
def euclidean_distance_mfcc(seq1, seq2):
    """ Euclidean distance with sliding window support """
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


# ====================== TEMPLATE MANAGEMENT ======================
def load_templates():
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
    print("\n" + "="*60)
    print("WEEK 4: RECORD TEMPLATES (using Divide-and-Conquer)")
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


# ====================== LIVE RECOGNITION ======================
def test_live_euclidean(use_dac=True):
    print("\n" + "="*60)
    print("WEEK 4: LIVE RECOGNITION (Euclidean + Divide-and-Conquer)")
    print("="*60)
    
    templates, template_names = load_templates()
    if len(templates) == 0:
        print("No templates found! Please run option 1 first.")
        return
    
    print("\nSpeak any command... (Ctrl+C to exit)\n")
    
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
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


# ====================== SIMULATED AUDIO TEST  ======================
def test_simulated(use_dac=True):
    """
    Simulated test with synthetic audio to demonstrate feature extraction and temolated matching without needing live recording.
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
    
    dc_features = divide_conquer_windowing(test_audio, 0, len(test_audio))
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
        
        # Find best match - FIXED
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
    
    # Recurrence Relation
    print("\n" + "-"*50)
    print("RECURRENCE RELATION FOR DIVIDE-AND-CONQUER WINDOWING")
    print("-"*50)
    print("T(n) = 2 * T(n/2) + Θ(1)")
    print("")
    print("Solution: T(n) = Θ(n log n)  (practical complexity)")
    print("This satisfies the Week 4 requirement of using Divide-and-Conquer for windowing.")


# ====================== DIVIDE-AND-CONQUER WINDOWING (for demo) ======================
def divide_conquer_windowing(audio, start, end, frame_size=256, features=None):
    """ Original Divide-and-Conquer for energy + ZCR (demo purpose) """
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
    divide_conquer_windowing(audio, start, mid, frame_size, features)
    divide_conquer_windowing(audio, mid, end, frame_size, features)
    return features


# ====================== MENU ======================
def print_menu():
    print("\n" + "="*60)
    print("   WEEK 4: TEMPLATE MATCHING FOUNDATION")
    print("="*60)
    for i, cmd in enumerate(commands, 1):
        print(f"   {i:2d}. {cmd}")
    print("\nOptions:")
    print("1. RECORD TEMPLATES (with Divide-and-Conquer)")
    print("2. LIVE RECOGNITION (Euclidean + DAC)")
    print("3. TEST SIMULATED AUDIO")
    print("4. Exit")


# ====================== MAIN ======================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   EMBEDDED SPEECH RECOGNITION SYSTEM - WEEK 4")
    print("="*60)
    
    while True:
        print_menu()
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            record_templates(use_dac=True)
        elif choice == "2":
            test_live_euclidean(use_dac=True)
        elif choice == "3":
            test_simulated(use_dac=True)
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice!")
        
        input("\nPress Enter to continue...")
