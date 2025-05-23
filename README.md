# EmoCeleb Dataset

The **EmoCeleb** dataset is a large-scale, weakly-labeled dataset for personalized emotion recognition, introduced in our paper **SetPeER: Set-based Personalized Emotion Recognition with Weak Supervision**. It is constructed from the VoxCeleb2 dataset using a cross-modal labeling approach, resulting in two subsets:

- **EmoCeleb-A**: Weak labels for speech modality derived from vision and text modalities.
- **EmoCeleb-V**: Weak labels for vision modality derived from speech and text modalities.

This dataset enables the development and evaluation of personalized emotion recognition systems with a large number of utterances per speaker.

## Repository Structure

```
├── LICENSE
├── README.md
├── data
│   ├── demographics_0216.csv
│   ├── labels_0216/              # Weak labels for EmoCeleb-A
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── val_test.csv
│   ├── labels_0330_vision/       # Weak labels for EmoCeleb-V
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── val_test.csv
│   ├── text_emotion_0116.csv     # Transcribed and inferred text emotion labels
│   ├── voxceleb2_inference.pkl   # Vision-based emotion logits
│   └── voxceleb2_wavlm_inference.pkl # Audio-based emotion logits
├── id_split.py                   # Script for speaker-level dataset splits
├── process.py                    # Script to derive EmoCeleb-A
├── process_vision.py            # Script to derive EmoCeleb-V
```

## Getting Started

### 1. Prerequisites

- Dependencies listed in `requirements.txt` 

### 2. VoxCeleb2 Download

To use this dataset, you must download the original **VoxCeleb2** videos. Please request access and download the dataset from the official website:

🔗 [https://mm.kaist.ac.kr/datasets/voxceleb/](https://mm.kaist.ac.kr/datasets/voxceleb/)

Ensure that the video directory structure is preserved after extraction.

### 3. Generating the Dataset

To regenerate the emotion label files:

- For **EmoCeleb-A** (audio modality):
  ```bash
  python process.py
  ```

- For **EmoCeleb-V** (visual modality):
  ```bash
  python process_vision.py
  ```

The scripts will generate per-utterance labels by aggregating and filtering the weak labels provided in the `data/` directory.

### 4. Dataset Format

Each CSV file (train/val/test) in `labels_0216` and `labels_0330_vision` contains:
- `utt_id`: Identifier of the utterance (matching VoxCeleb2 file IDs)
- `label`: Weakly predicted emotion label (`neutral`, `happiness`, `anger`, `surprise`)
- `speaker_id`: Unique speaker identifier
- `gender`: gender of the speaker, provided by VoxCeleb2
- `ethnicity`: ethnicity of the speaker, generated by prompting LLM

### 5. Citation

If you use this dataset, please cite:

```
@article{tran2025setpeer,
  title={SetPeER: Set-based Personalized Emotion Recognition with Weak Supervision},
  author={Tran, Minh and Yin, Yufeng and Soleymani, Mohammad},
  journal={IEEE Transactions on Affective Computing},
  year={2025}
}
```