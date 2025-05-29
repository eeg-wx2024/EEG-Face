# EEG-Face Analysis Project

This project focuses on analyzing the EEG-Face dataset to investigate neural correlations, super-trial EEG, EEG-stimulus pairing, and face recognition.

*(Note: The original text mentions "The overall workflow and components are illustrated in the figure above." Please ensure this figure is embedded or linked in your actual README file where appropriate, for example, by adding an image link like `![Workflow Figure](path/to/your/figure.png)`)*

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

*(Ensure that your `requirements.txt` file is comprehensive and includes all libraries and their specific versions needed to run the code, e.g., PyTorch, Scikit-learn, MNE-Python, Pandas, NumPy, etc.)*

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

**Results Summary:** *Table:* Gender identification results *via classifying randomized EEG trials*

| Model                                                   | Accuracy |
| ------------------------------------------------------- | -------- |
| SVM \cite{li2020perils}                                 | 52.3%    |
| KNN \cite{li2020perils}                                 | 51.2%    |
| LSTM \cite{hochreiter1997lstm}                          | 54.3%    |
| MLP \cite{li2020perils}                                 | 53.8%    |
| CNN \cite{li2020perils}                                 | 54.1%    |
| SyncNet \cite{li2017targeting}                          | 54.0%    |
| EEGChannelNet \cite{palazzo2020decoding}                | 54.1%    |
| EEGConformer \cite{song2022eeg}                         | 54.0%    |
| NICE-EEG \cite{song2023decoding}                        | 54.1%    |
| EEGNet \cite{lawhern2018eegnet}                         | 54.4%    |
| EEGNet \cite{lawhern2018eegnet} (10 males & 20 females) | 68.8%    |

*(Note: The original table format was a continuous string; it has been parsed into a two-column table for clarity.)*

### b) EEG-Stimulus Pairing Verification

This process uses an Image Encoder and an EEG Encoder, possibly with a contrastive approach, to verify the match between EEG signals and the stimuli presented. It performs 1:1 and 1:N verification during testing.

**To run this analysis, execute:**

```
# python your_script_for_eeg_stimulus_pairing.py # Placeholder: The command was not specified in the original text. Please update this line with the correct script.
```

**(Important: The execution command for this specific analysis was not provided in your original text. Please add the correct Python script to run this part.)**

**Results Summary:** *Table: Results of EEG-Image Pairings* (Original LaTeX table `\label{tab:pairing-results}` converted to Markdown)

| Trial Setting | 1:1 Verification (Top-1) | 1:N Verification (Top-1) | 1:N Verification (Top-5) |
| ------------- | ------------------------ | ------------------------ | ------------------------ |
| Single-trial  | 58.0%                    | 3.9%                     | 17.5%                    |
| Super-trial   | 67.0%                    | 4.4%                     | 20.5%                    |

### c) Face Recognition via Classification of Randomized EEG Trials

This involves classifying randomized EEG trials (both single-trial and super-trial, though the table focuses on randomized trials) to perform face recognition and categorize them.

**To run this analysis, execute:**

```
python main_model.py
```

**Results Summary:** *Table: Results of brain-perceived face recognition (Randomized EEG Trials)* (Original LaTeX table `\label{table:Experimental Results}` converted and simplified for Markdown)

| Model                                    | Top-1 Accuracy | Top-5 Accuracy | Params    |
| ---------------------------------------- | -------------- | -------------- | --------- |
| SVM \cite{li2020perils}                  | 3.2%           | 13.3%          | 1.9M      |
| KNN \cite{li2020perils}                  | 2.7%           | 11.8%          | -         |
| LSTM \cite{hochreiter1997lstm}           | 3.3%           | 14.1%          | 0.1M      |
| MLP \cite{li2020perils}                  | 3.1%           | 13.5%          | 6.3M      |
| CNN \cite{li2020perils}                  | 3.4%           | 14.4%          | 0.004M    |
| EEGNet \cite{lawhern2018eegnet}          | 3.4%           | 14.7%          | 0.01M     |
| EEGChannelNet \cite{palazzo2020decoding} | 3.2%           | 14.2%          | 4.5M      |
| SyncNet \cite{li2017targeting}           | 3.4%           | 14.8%          | 0.04M     |
| EEGConformer \cite{song2022eeg}          | 3.3%           | 14.1%          | 0.6M      |
| NICE-EEG \cite{song2023decoding}         | 3.2%           | 14.1%          | 0.3M      |
| **EEG-TSSnet (Proposed)**                | **15.1%**      | **39.7%**      | **0.05M** |

*(Note:* The original LaTeX table was *complex and included columns for "Deterministic EEG trials" which were empty in the provided snippet. This Markdown version simplifies the presentation focusing on the provided "Randomized EEG Trials" data.)*

### d) Super-trial EEG Analysis

This analysis likely refers to evaluations performed on aggregated "Super-Trial" EEG data. The script `main_model-10to1.py` suggests a configuration, possibly where 10 single trials are combined into one super-trial, for tasks like EEG-Image pairings or classification.

The original text also mentions a file: `super_top1_top5.pdf`. This PDF likely contains graphical representations of results (e.g., Top-1 and Top-5 accuracies) for this super-trial analysis. Ensure this file is accessible in your project repository or linked appropriately.

**To run this analysis (e.g., for a 10-to-1 super-trial setup):**

```
python main_model-10to1.py
```

## References (from table citations)

The citations like `\cite{li2020perils}` refer to academic papers. You should include a full bibliography or a list of references in your project documentation or paper. For example:

- `li2020perils`: [Full citation details for Li et al., 2020]
- `palazzo2020decoding`: [Full citation details for Palazzo et al., 2020]
- `hochreiter1997lstm`: [Full citation details for Hochreiter & Schmidhuber, 1997]
- `lawhern2018eegnet`: [Full citation details for Lawhern et al., 2018]
- `song2022eeg`: [Full citation details for Song et al., 2022]
- `song2023decoding`: [Full citation details for Song et al., 2023]
- `li2017targeting`: [Full citation details for Li et al., 2017]

*(Please replace the bracketed placeholders with the actual bibliographic information for these papers.)*