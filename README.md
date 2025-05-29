# EEG-Face Analysis Project

This project focuses on analyzing the EEG-Face dataset to investigate neural correlations, super-trial EEG, EEG-stimulus pairing, and face recognition.

![fig1](E:\0_ACM-MM人脸数据集\图\fig1.png)

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset: EEG-Face](#dataset-eeg-face)
3. [Code Structure and Analyses](#code-structure-and-analyses)
   - [a)](#a-neural-correlation-of-gender-perceptions) Neural Correlation of Gender Perceptions
   - [b) EEG-Stimulus Pairing Verification](#b-eeg-stimulus-pairing-verification)
   - [c) Face Recognition via Classification of Randomized EEG Trials](#c-face-recognition-via-classification-of-randomized-eeg-trials)
   - [d) Super-trial EEG Analysis](#d-super-trial-eeg-analysis)
4. [References (from table citations)](#references-from-table-citations)

## Environment Setup

To set up the project environment and install the necessary dependencies, run the following command in your terminal:

```
pip install -r requirements.txt
```

## Dataset: EEG-Face

The dataset used in this project is the EEG-Face dataset.

- **Content:** It contains EEG signals collected from 40 participants, where each participant viewed 500 image stimuli. This results in a total of 20,000 EEG signals.
- **Image Processing:** The images underwent a "Content-rich Expansion" phase using "PhotoMaker" before being used in the EEG-Face Collection.
- **Trial Types:** EEG signals are available as single trials and can also be aggregated into super-trials for analysis.
- **Download:** The dataset can be downloaded from: https://osf.io/s2jyx/

## Code Structure and Analyses

The project performs several verification analyses on the collected EEG-Face dataset. The main scripts and their purposes are outlined below:

### a) Neural Correlation of Gender Perceptions

This analysis involves analyzing EEG signals to understand the neural basis of gender perception.

**To run this analysis, execute:**

```
python main_model-sex.py
```

### b) EEG-Stimulus Pairing Verification

This process uses an Image Encoder and an EEG Encoder, possibly with a contrastive approach, to verify the match between EEG signals and the stimuli presented. It performs 1:1 and 1:N verification during testing.

**To run this analysis, execute:**

```
 python EEG-Image_pairings.py 
```

### c) Face Recognition via Classification of Randomized EEG Trials

This involves classifying randomized EEG trials (both single-trial and super-trial, though the table focuses on randomized trials) to perform face recognition and categorize them.

**To run this analysis, execute:**

```
python main_model.py
```

### d) Varying the number of identities from 2 to 40

To enable an in-depth evaluation of EEG-Face, we select the well-performed EEGNet as a representative and test its face recognition performances by varying the number of identities from 2 to 40.

**To run this analysis, execute:**

```
python main_eeg-2-40.py
```

### e) Super-trial EEG Analysis

Evaluation on “super trial” EEG data. Ten single trials are combined into a super trial for tasks such as EEG-image classification.

**To run this analysis, execute:**

```
python main_model-10to1.py
```