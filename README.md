# Traffic Sign Recognition with Deep Learning

Welcome to the **Traffic Sign Recognition** project! This repository implements a deep learning solution for classifying traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Leveraging Convolutional Neural Networks (CNNs) and transfer learning with MobileNetV2, this project supports autonomous driving and advanced driver-assistance systems (ADAS) by recognizing 43 traffic sign classes.

## Project Overview

This project develops and evaluates deep learning models for traffic sign classification. Key objectives include:
- Preprocessing images (resizing, normalization).
- Building a custom CNN for multi-class classification.
- Applying data augmentation to improve generalization.
- Training and evaluating models with accuracy and confusion matrices.
- Comparing performance across custom CNN (with and without augmentation) and MobileNetV2.

### Key Results
- **Custom CNN (No Augmentation)**: **95.72%** test accuracy.
- **Augmented Custom CNN**: **92.09%** test accuracy.
- **MobileNetV2**: **85.14%** test accuracy (improved with fine-tuning).
- **Fine-tuned MobileNetV2**: Achieves higher validation accuracy (see plots).

## Repository Structure

```
traffic-sign-recognition/
├── notebooks/
│   └── traffic-sign-recognition.ipynb  # Jupyter notebook with project code
├── plots/
│   ├── custom_cnn_accuracy_loss.png   # Accuracy and loss plots for Custom CNN
│   ├── augmented_cnn_accuracy_loss.png # Accuracy and loss plots for Augmented CNN
│   ├── mobilenet_accuracy_loss.png    # Accuracy and loss plots for MobileNet
│   ├── finetuned_mobilenet_accuracy_loss.png # Accuracy and loss plots for Fine-tuned MobileNet
│   ├── custom_cnn_confusion_matrix.png # Confusion matrix for Custom CNN
│   ├── augmented_cnn_confusion_matrix.png # Confusion matrix for Augmented CNN
│   ├── mobilenet_confusion_matrix.png  # Confusion matrix for MobileNet
│   └── sample_traffic_signs.png        # Sample visualizations of traffic signs
├── requirements.txt                    # Project dependencies
└── README.md                          # Project overview and setup instructions
```

## Prerequisites

- **Python**: Version 3.11.13
- **Dataset**: GTSRB dataset (available on [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign))
- **Hardware**: GPU (e.g., NVIDIA Tesla T4) recommended for faster training

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Abdelrahman2610/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   - Download the GTSRB dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
   - Extract it to a directory (e.g., `data/`) and update the `base_path` variable in the notebook to point to this directory.

## Usage

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Run the Notebook**:
   - Open `notebooks/traffic-sign-recognition.ipynb`.
   - Execute the cells to load data, preprocess images, train models, and evaluate results.

3. **Notebook Sections**:
   - **Introduction**: Project and dataset overview.
   - **Data Loading**: Loads and visualizes GTSRB data.
   - **Data Preprocessing**: Resizes and normalizes images.
   - **Model Building**: Defines custom CNN and MobileNetV2 architectures.
   - **Model Training**: Trains models with and without augmentation.
   - **Model Evaluation**: Assesses performance with accuracy and confusion matrices.
   - **Model Comparison**: Compares results across models.

4. **Visualizations**:
   - Training and validation plots are saved in `plots/` (e.g., `custom_cnn_accuracy_loss.png`).
   - Confusion matrices are saved as `custom_cnn_confusion_matrix.png`, etc.

## Results

### Training and Validation Plots
- **Custom CNN**:
  ![Custom CNN Accuracy and Loss](plots/custom_cnn_accuracy_loss.png)
- **Augmented Custom CNN**:
  ![Augmented CNN Accuracy and Loss](plots/augmented_cnn_accuracy_loss.png)
- **MobileNet**:
  ![MobileNet Accuracy and Loss](plots/mobilenet_accuracy_loss.png)
- **Fine-tuned MobileNet**:
  ![Fine-tuned MobileNet Accuracy and Loss](plots/finetuned_mobilenet_accuracy_loss.png)

### Confusion Matrices
- **Custom CNN**:
  ![Custom CNN Confusion Matrix](plots/custom_cnn_confusion_matrix.png)
- **Augmented Custom CNN**:
  ![Augmented CNN Confusion Matrix](plots/augmented_cnn_confusion_matrix.png)
- **MobileNet**:
  ![MobileNet Confusion Matrix](plots/mobilenet_confusion_matrix.png)

These plots and matrices demonstrate model performance, with the Custom CNN achieving the highest test accuracy (95.72%) and the Augmented CNN showing improved generalization.

## Kaggle Notebook

For the original implementation and detailed notebook, visit my Kaggle profile:  
[Traffic Sign Recognition on Kaggle](https://www.kaggle.com/code/abdelrahmansalah2002/traffic-sign-recognition)

## Future Improvements

- Fine-tune MobileNetV2 further to close the accuracy gap.
- Add detailed analysis of confusion matrix errors to identify misclassifications.
- Enhance visualizations with captions and optimized layouts.
- Include a table of contents in the notebook for better navigation.

## References

- GTSRB Dataset: [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- TensorFlow Documentation: [tensorflow.org](https://www.tensorflow.org)
- Keras Documentation: [keras.io](https://keras.io)

## Contributing

Contributions are welcome! Submit pull requests or open issues for bugs, enhancements, or new features.
