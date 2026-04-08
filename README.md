# EMBEDDED VOICE RECOGNITION SYSTEM 
# Group 4 Members

MWAMBU ISAIAH 2401600395

KOBUSINGYE RITAH 2401600224

ATIM ANNA MARY 2401600079

KABWOKO ENOCH 2401600225

  
# Group 4 Week 4 Deliverable: Template Matching Foundation

## Project Overview

This is a **prototyping version** of an Embedded Speech Recognition System designed for a low-power smart home assistant targeted at elderly users. The system can recognize **10 voice commands** using template matching.

### Supported Commands
- turn light on  
- turn light off  
- increase temperature  
- decrease temperature  
- turn fan on  
- turn fan off  
- open door  
- close door  
- open window  
- close window  

## Key Features & Technologies

**Sampling Rate**: 8 kHz (standard for embedded speech)
**Feature Extraction**: 13 MFCC coefficients
**Voice Activity Detection (VAD)**: Energy-based silence trimming
**Template Matching**: Euclidean distance with sliding window support
**Windowing Strategy**: Divide-and-Conquer (recursive) approach for feature extraction
**Testing Modes**:
Live microphone input with real-time recognition
Simulated audio test for offline validation
**Recurrence Relation Analysis**: Included for Divide-and-Conquer windowing


## System Architecture

Records audio at 8 kHz using PC microphone
Applies VAD to remove silence
Extracts features using Divide-and-Conquer windowing + MFCC
Compares the unknown utterance with stored templates using Euclidean distance
Displays the recognized command with distance visualization

## How to Run

1. Run the program:
   ```bash
   python mffc.py

<img width="840" height="316" alt="image" src="https://github.com/user-attachments/assets/a5f1ae1b-b32f-45b4-bb03-9268c8ce096f" />
