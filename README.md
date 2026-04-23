# EMBEDDED VOICE RECOGNITION SYSTEM 
# Group 4 Members

MWAMBU ISAIAH 2401600395

KOBUSINGYE RITAH 2401600224

ATIM ANNA MARY 2401600079

KABWOKO ENOCH 2401600225

  

## Project Overview
This is a **complete offline speech recognition system** designed for smart home automation targeting elderly users. The system recognizes **10 voice commands** using template matching with both **Euclidean distance**  and **Dynamic Time Warping (DTW)** algorithms.


### Key Features

**Offline Operation**  No internet required - runs completely locally 
**Real-time Recognition**  Processes speech as you speak 
**WAV File Support**  Test with pre-recorded audio files 
**DTW Algorithm**  Handles different speaking speeds 
**Memory Optimized**  DTW uses only 2 rows (95% memory saving) 
**Accuracy Comparison**  Euclidean vs DTW side-by-side 
**Trade-off Analysis**  Template size, fixed vs float analysis 



##  System Requirements

### Hardware (Minimum)
- Computer with microphone (for live recognition)

### Software
- Python 3.8 or higher
- Required libraries (see Installation)


## 🗣️ Supported Commands

The system recognizes commonly used **10 smart home commands**:

| Category | Commands |
|----------|----------|
|  **Lighting** | "turn light on", "turn light off" |
|  **Temperature** | "increase temperature", "decrease temperature" |
|  **Fans** | "turn fan on", "turn fan off" |
|  **Doors** | "open door", "close door" |
|  **Windows** | "open window", "close window" |



##  Technical Specifications

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Sampling Rate** | 8 kHz | Phone-quality audio, sufficient for speech |
| **Frame Size** | 256 samples | 32ms windows - balances time/frequency resolution |
| **Frame Shift** | 128 samples | 50% overlap for smooth transitions |
| **MFCC Coefficients** | 13 | Industry standard for speech recognition |
| **Mel Filters** | 40 | Standard for MFCC computation |
| **Audio Format** | Mono, 16-bit PCM |  for embedded systems |

---

##  Algorithms Implemented

### Week 4: Euclidean Distance
Distance = √( Σ (aᵢ - bᵢ)² )


### Week 6: Dynamic Time Warping (DTW)
D[i][j] = cost(i,j) + min(D[i-1][j], D[i][j-1], D[i-1][j-1])

text
- Handles variations in speaking speed
- Memory-optimized: only stores 2 rows (O(m) space)
- Improved accuracy

### Divide-and-Conquer Windowing
T(n) = 2T(n/2) + Θ(1) → Θ(n log n)

text
- Recursive approach for feature extraction
- Demonstrates algorithmic thinking



### Data Flow Explanation

1. **Audio Input**: From microphone OR WAV file
2. **Voice Activity Detection (VAD)**: Removes silence using energy threshold
3. **Feature Extraction**: MFCC with Divide-and-Conquer windowing
4. **Template Matching**: Compare against stored templates
5. **Recognition Output**: Best matching command displayed


 ### How to Use
First Time Setup (Recording Templates)
text
1. Run: python final-code.py
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
 Features in Detail




. Fixed Point vs Floating Point
Aspect	Floating Point	Fixed Point (Q15)
Precision	High (24-bit)	Medium (15-bit)
Memory	4 bytes/value	2 bytes/value
Speed	Requires FPU	Fast on any CPU
Ease of coding	Easy	Complex
Recommendation: Use floating point for development, fixed point for ESP32 deployment


# Troubleshooting
Common Issues and Solutions
Issue	Possible Cause	Solution
"No sounddevice"	Library not installed	pip install sounddevice
"No speech detected"	Microphone not working	Check mic permissions
"PortAudioError"	Audio device issue	Try different USB port
Low accuracy	Poor template recording	Re-record templates clearly
DTW too slow	Too many templates	Reduce to 1-2 per command
WAV file not found	Wrong directory	Place WAV in same folder
Microphone Not Working?



### Future Improvements
Port to C for ESP32 deployment

Add more templates (3-5 per command)

Implement Gaussian Mixture Models (GMM)

Add noise reduction preprocessing

Create mobile app interface

Add wake word detection ("Hey Assistant")

