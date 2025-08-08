# PPG Blood Pressure Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Overview

A state-of-the-art non-invasive blood pressure prediction system based on Photoplethysmography (PPG) signals. This project combines deep learning and traditional machine learning approaches to achieve accurate prediction of Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP). The system integrates advanced signal processing, feature engineering, contrastive learning, and adaptive feature space techniques.

### 🎯 Key Features

- **Multi-modal Signal Processing**: Advanced PPG signal filtering, denoising, and normalization
- **Intelligent Feature Extraction**: Automated extraction of time-domain, frequency-domain, and nonlinear dynamics features
- **Contrastive Learning Framework**: Self-supervised learning with SimCLR, SupCon, and other methods
- **Adaptive Feature Space**: Dynamic feature representation adjustment for individual differences
- **Multi-model Ensemble**: Integration of traditional ML and deep learning models
- **Interactive Visualization**: Rich data and result visualization tools
- **Real-time Prediction**: Support for real-time PPG signal blood pressure prediction

## 🏗️ System Architecture

```
PPG Blood Pressure Prediction System
├── Data Preprocessing Module
│   ├── Signal Filtering & Denoising
│   ├── Quality Assessment
│   └── Data Augmentation
├── Feature Engineering Module
│   ├── Time-domain Feature Extraction
│   ├── Frequency-domain Feature Extraction
│   ├── Nonlinear Feature Extraction
│   └── Adaptive Feature Selection
├── Contrastive Learning Module
│   ├── SimCLR Contrastive Learning
│   ├── Supervised Contrastive Learning
│   └── Triplet Loss Learning
├── Model Training Module
│   ├── Traditional Machine Learning Models
│   ├── Deep Neural Networks
│   └── Ensemble Learning Methods
├── Adaptive Space Module
│   ├── Neural Network Transformer
│   ├── Feature Space Clustering
│   └── Space Quality Assessment
└── Visualization Analysis Module
    ├── Signal Visualization
    ├── Feature Analysis
    ├── Model Performance Evaluation
    └── Interactive Charts
```

## 📦 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Quick Install

```bash
# Clone repository
git clone https://github.com/your-username/ppg-bp-prediction.git
cd ppg-bp-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

### Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.7.0
joblib>=1.0.0
tqdm>=4.62.0
```

## 🚀 Quick Start

### 1. Data Preparation

```python
from src.data.preprocessing import PPGPreprocessor
from src.data.data_loader import PPGDataLoader

# Load data
data_loader = PPGDataLoader(config)
ppg_signals, bp_labels = data_loader.load_data("path/to/your/data")

# Preprocessing
preprocessor = PPGPreprocessor(config)
processed_signals = preprocessor.preprocess_batch(ppg_signals)
```

### 2. Feature Extraction

```python
from src.features.feature_extraction import FeatureExtractor

# Initialize feature extractor
extractor = FeatureExtractor(config)

# Extract features
features = []
for signal in processed_signals:
    feature_dict = extractor.extract_all_features(signal)
    feature_vector = extractor.features_to_vector(feature_dict)
    features.append(feature_vector)

features = np.array(features)
```

### 3. Model Training

```python
from src.models.traditional_models import TraditionalModelManager
from src.training.contrastive_learning import ContrastiveTrainer

# Traditional machine learning models
model_manager = TraditionalModelManager(config)
model_manager.train_all_models(X_train, y_train)

# Contrastive learning
contrastive_trainer = ContrastiveTrainer(config)
train_loader, val_loader, test_loader = create_contrastive_dataloaders(
    features, bp_labels, config
)
contrastive_trainer.train(train_loader, val_loader, epochs=100)
```

### 4. Results Visualization

```python
from src.utils.visualization import ModelVisualizer, create_comprehensive_report

# Evaluate models
evaluation_results = model_manager.evaluate_all_models(X_test, y_test)

# Generate comprehensive report
create_comprehensive_report(
    y_test, y_pred, 
    feature_importance=feature_importance,
    feature_names=feature_names,
    model_name="RandomForest",
    save_dir="results/"
)
```


## 🛠️ API Documentation

### Core APIs

```python
# Data preprocessing
preprocessor = PPGPreprocessor(config)
clean_signal = preprocessor.preprocess_single(raw_signal)

# Feature extraction
extractor = FeatureExtractor(config)
features = extractor.extract_all_features(clean_signal)

# Model prediction
predictor = BPPredictor(model_path)
bp_prediction = predictor.predict(features)

# Result visualization
visualizer = PPGVisualizer(config)
visualizer.plot_ppg_signal(clean_signal)
```

### Configuration Parameters

```python
config = {
    # Data processing parameters
    'sampling_rate': 125,
    'filter_order': 4,
    'lowcut': 0.5,
    'highcut': 8.0,
    
    # Feature extraction parameters
    'window_size': 1000,
    'overlap': 0.5,
    'feature_types': ['time', 'frequency', 'nonlinear'],
    
    # Model training parameters
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 10,
    
    # Contrastive learning parameters
    'temperature': 0.07,
    'embedding_dim': 128,
    'augmentation_strength': 0.5
}
```

## 📚 Documentation

- [Data Preprocessing Guide](docs/preprocessing.md)
- [Feature Engineering Details](docs/feature_engineering.md)
- [Model Training Tutorial](docs/training.md)
- [API Reference Manual](docs/api_reference.md)
- [Performance Optimization Guide](docs/optimization.md)

## 🤝 Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to participate in project development.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to [PhysioNet](https://physionet.org/) for providing open datasets
- Thanks to [MIMIC-III](https://mimic.mit.edu/) database support
- Thanks to all contributors for their hard work

## 📞 Contact

- **Project Maintainer**: [Your Name](mailto:your.email@example.com)
- **Project Homepage**: [https://github.com/your-username/ppg-bp-prediction](https://github.com/your-username/ppg-bp-prediction)
- **Issue Reports**: [Issues](https://github.com/your-username/ppg-bp-prediction/issues)
- **Discussions**: [Discussions](https://github.com/your-username/ppg-bp-prediction/discussions)

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/your-username/ppg-bp-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/ppg-bp-prediction?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/ppg-bp-prediction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/your-username/ppg-bp-prediction)

---

**If this project helps you, please give us a ⭐️!**

```
