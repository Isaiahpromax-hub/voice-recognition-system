#  EMBEDDED SPEECH RECOGNITION SYSTEM

## Group 4( Final Project)

## Group Members



---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Requirements](#system-requirements)
- [Supported Commands](#supported-commands)
- [Technical Specifications](#technical-specifications)
- [Algorithms Implemented](#algorithms-implemented)
- [System Architecture](#system-architecture)
- [Installation Guide](#installation-guide)
- [How to Use](#how-to-use)
- [Features in Detail](#features-in-detail)
- [Accuracy Results](#accuracy-results)
- [Complexity Analysis](#complexity-analysis)
- [Trade-off Analysis](#trade-off-analysis)
- [Deliverables Checklist](#deliverables-checklist)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🎯 Project Overview

This is a **complete offline speech recognition system** designed for smart home automation targeting elderly users. The system recognizes **10 voice commands** using template matching with both **Euclidean distance** (Week 4) and **Dynamic Time Warping (DTW)** (Week 6) algorithms.

### Key Features

| Feature | Description |
|---------|-------------|
| 🔒 **Offline Operation** | No internet required - runs completely locally |
| 🎙️ **Real-time Recognition** | Processes speech as you speak |
| 📁 **WAV File Support** | Test with pre-recorded audio files |
| 🧠 **DTW Algorithm** | Handles different speaking speeds |
| 💾 **Memory Optimized** | DTW uses only 2 rows (95% memory saving) |
| 📊 **Accuracy Comparison** | Euclidean vs DTW side-by-side |
| 🔧 **Trade-off Analysis** | Template size, fixed vs float analysis |

---

## 💻 System Requirements

### Hardware (Minimum)
- Computer with microphone (for live recognition)
- OR any device that can run Python 3.8+

### Software
- Python 3.8 or higher
- Required libraries (see Installation)

---

## 🗣️ Supported Commands

The system recognizes **10 smart home commands**:

| Category | Commands |
|----------|----------|
| 💡 **Lighting** | "turn light on", "turn light off" |
| 🌡️ **Temperature** | "increase temperature", "decrease temperature" |
| 🌀 **Fans** | "turn fan on", "turn fan off" |
| 🚪 **Doors** | "open door", "close door" |
| 🪟 **Windows** | "open window", "close window" |

---

## ⚙️ Technical Specifications

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Sampling Rate** | 8 kHz | Phone-quality audio, sufficient for speech |
| **Frame Size** | 256 samples | 32ms windows - balances time/frequency resolution |
| **Frame Shift** | 128 samples | 50% overlap for smooth transitions |
| **MFCC Coefficients** | 13 | Industry standard for speech recognition |
| **Mel Filters** | 40 | Standard for MFCC computation |
| **Audio Format** | Mono, 16-bit PCM | Suitable for embedded systems |

---

## 🧮 Algorithms Implemented

### Week 4: Euclidean Distance
Distance = √( Σ (aᵢ - bᵢ)² )

text
- Simple, fast O(n) computation
- Sliding window support for different lengths
- Baseline accuracy: ~70%

### Week 6: Dynamic Time Warping (DTW)
D[i][j] = cost(i,j) + min(D[i-1][j], D[i][j-1], D[i-1][j-1])

text
- Handles variations in speaking speed
- Memory-optimized: only stores 2 rows (O(m) space)
- Improved accuracy: ~90%

### Divide-and-Conquer Windowing
T(n) = 2T(n/2) + Θ(1) → Θ(n log n)

text
- Recursive approach for feature extraction
- Demonstrates algorithmic thinking

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────────────┐
│ SPEECH RECOGNITION SYSTEM │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ INPUT PROCESSING OUTPUT │
│ │
│ ┌─────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │Microphone│───────────→│ VAD │───────────→│ Silence │ │
│ │ │ │ (Energy) │ │ Removed │ │
│ └─────────┘ └──────┬──────┘ └──────┬──────┘ │
│ │ │ │
│ ┌─────────┐ ↓ ↓ │
│ │ WAV File│ ┌─────────────┐ ┌─────────────┐ │
│ │ Reader │───────────→│ MFCC │───────────→│ Features │ │
│ └─────────┘ │ Extraction │ │ (13×frames)│ │
│ └──────┬──────┘ └──────┬──────┘ │
│ │ │ │
│ ↓ ↓ │
│ ┌─────────────┐ ┌─────────────┐ │
│ │ Template │ │ Distance │ │
│ │ Matching │←───────────│ Matrix │ │
│ │ (DTW/Euc) │ │ │ │
│ └──────┬──────┘ └─────────────┘ │
│ │ │
│ ↓ │
│ ┌─────────────┐ │
│ │ RECOGNIZED │ │
│ │ "turn light │ │
│ │ on" │ │
│ └─────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────┘

text

### Data Flow Explanation

1. **Audio Input**: From microphone OR WAV file
2. **Voice Activity Detection (VAD)**: Removes silence using energy threshold
3. **Feature Extraction**: MFCC with Divide-and-Conquer windowing
4. **Template Matching**: Compare against stored templates
5. **Recognition Output**: Best matching command displayed

---

## 📦 Installation Guide

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/yourusername/speech-recognition-system.git
cd speech-recognition-system
Step 2: Install Required Libraries
bash
pip install numpy sounddevice librosa
Or use the requirements file:

bash
pip install -r requirements.txt
Step 3: Verify Installation
bash
python -c "import numpy, sounddevice, librosa; print('All libraries installed successfully!')"
Step 4: Run the System
bash
python mffc.py
🎮 How to Use
First Time Setup (Recording Templates)
text
1. Run: python mffc.py
2. Select Option 1: RECORD TEMPLATES
3. Speak each command clearly when prompted
4. Repeat for all 10 commands
Live Recognition
text
Option 2: EUCLIDEAN - Faster but less accurate
Option 3: DTW - More accurate, handles speed variations
Testing Options
Option	Function	Best For
1	Record Templates	Initial setup
2	Live Euclidean	Quick testing
3	Live DTW	Accurate recognition
4	Compare Algorithms	Accuracy measurement
5	Simulated Audio	Debugging
6	WAV File Test	Batch testing
7	Complexity Report	Understanding performance
8	Fixed vs Float Demo	Embedded considerations
9	Exit	Quit program
✨ Features in Detail
1. Voice Activity Detection (VAD)
python
Energy = Σ(sample²) / frame_length
If Energy > 0.005 → Speech
If Energy < 0.005 → Silence
Automatically trims silence from recordings

Improves accuracy by focusing on speech only

2. MFCC Feature Extraction
Converts raw audio to 13 coefficients per frame

Mimics human ear frequency perception

Creates unique "fingerprint" for each sound

3. Divide-and-Conquer Windowing
Recursive algorithm for frame positioning

Demonstrates algorithmic efficiency

Complexity: Θ(n log n)

4. DTW with Memory Optimization
Only stores 2 rows instead of full matrix

Memory saving: 95% reduction

Maintains same accuracy as standard DTW

5. WAV File Support
Read standard 16-bit PCM WAV files

Automatic resampling to 8kHz

Stereo to mono conversion

📊 Accuracy Results
Test Methodology
10 test samples (one per command)

Same speaker for templates and tests

1 template per command (baseline)

Results Table
Command	Euclidean	DTW
turn light on	✓	✓
turn light off	✗	✓
increase temperature	✓	✓
decrease temperature	✗	✓
turn fan on	✓	✓
turn fan off	✗	✗
open door	✓	✓
close door	✓	✓
open window	✗	✓
close window	✓	✓
TOTAL	7/10	9/10
ACCURACY	70%	90%
Key Finding
DTW improves accuracy by 20% by handling variations in speaking speed.

⏱️ Complexity Analysis
DTW Complexity
Metric	Standard DTW	Our Optimized DTW
Time	O(n × m)	O(n × m)
Space	O(n × m)	O(m) ✓
Memory for n=m=20	400 cells	20 cells
Feature Extraction Complexity
text
Divide-and-Conquer:  T(n) = 2T(n/2) + Θ(1)  →  Θ(n log n)
MFCC per frame:      O(F log F) where F = 256
Overall:             O(N log N) where N = audio samples
Real-time Performance
text
For 2-second utterance (16000 samples):
- VAD: O(N) = 16,000 ops
- Feature extraction: O(N log N) ≈ 224,000 ops
- DTW matching (30 templates): ≈ 12,000 ops
- TOTAL: ≈ 252,000 operations

Estimated time on ESP32 (240MHz): ~30ms → Real-time capable!
⚖️ Trade-off Analysis
1. Template Size vs Accuracy
Templates per Command	Accuracy	Recognition Time
1	70-80%	Fastest
3	85-90%	Medium
5	90-95%	Slower
Recommendation: 3 templates per command for optimal balance

2. Fixed Point vs Floating Point
Aspect	Floating Point	Fixed Point (Q15)
Precision	High (24-bit)	Medium (15-bit)
Memory	4 bytes/value	2 bytes/value
Speed	Requires FPU	Fast on any CPU
Ease of coding	Easy	Complex
Recommendation: Use floating point for development, fixed point for ESP32 deployment

3. Frame Size vs Responsiveness
Frame Size	Time Resolution	Frequency Resolution	Best For
128 samples	Better	Worse	Fast speech
256 samples	Balanced	Balanced	General use ✓
512 samples	Worse	Better	Slow speech
✅ Deliverables Checklist
Week 4 Deliverables
Euclidean distance with sliding window

Divide-and-Conquer for feature extraction

MFCC extraction (13 coeffs, 8kHz)

Voice Activity Detection

Simulated audio test

Recurrence relation analysis

Week 6 Deliverables
DTW DP algorithm for sequence alignment

Backtracking for path reconstruction

Comparison with Euclidean distance

Real-time recognition code

DP recurrence documentation

Complexity analysis O(n×m)

Memory optimization (2 rows only)

Final System Requirements
Microphone input integration

WAV file reader

Real-time recognition

Data structures (Circular Buffer)

Complexity analysis (DTW + Feature extraction)

Accuracy testing on recorded samples

Trade-off analysis (template size vs accuracy)

Fixed point vs floating point comparison

🐛 Troubleshooting
Common Issues and Solutions
Issue	Possible Cause	Solution
"No sounddevice"	Library not installed	pip install sounddevice
"No speech detected"	Microphone not working	Check mic permissions
"PortAudioError"	Audio device issue	Try different USB port
Low accuracy	Poor template recording	Re-record templates clearly
DTW too slow	Too many templates	Reduce to 1-2 per command
WAV file not found	Wrong directory	Place WAV in same folder
Microphone Not Working?
python
# Test microphone with this quick script
import sounddevice as sd
print(sd.query_devices())  # Lists available devices
Poor Recognition Accuracy?
Re-record templates in a quiet environment

Speak clearly at normal volume

Use DTW instead of Euclidean (Option 3)

Record 3 templates per command

🚀 Future Improvements
Port to C for ESP32 deployment

Add more templates (3-5 per command)

Implement Gaussian Mixture Models (GMM)

Add noise reduction preprocessing

Create mobile app interface

Add wake word detection ("Hey Assistant")

