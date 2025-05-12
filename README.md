# US Currency Classification

A deep learning project for classifying US currency denominations using computer vision and neural networks.

![Currency Classification](output.......png)

## Project Overview

This project uses deep learning techniques to classify images of US currency denominations. The system can identify six different US dollar bills:
- 1 Dollar
- 2 Dollar
- 5 Dollar
- 10 Dollar
- 50 Dollar
- 100 Dollar

The project implements two different approaches:
1. A TensorFlow/Keras implementation using ResNet50
2. A PyTorch implementation with Particle Swarm Optimization (PSO) for hyperparameter tuning

## Dataset

The dataset consists of images of US currency organized in the following structure:
```
USA currency data/
├── 1 Dollar/
├── 2 Dollar/
├── 5 Dollar/
├── 10 Dollar/
├── 50 Dollar/
├── 100 Dollar/
```

Each directory contains multiple images of the corresponding denomination from different angles and lighting conditions.

## Features

- **Data Preprocessing**: Cleaning corrupted images, resizing, normalization
- **Transfer Learning**: Using pre-trained ResNet50 model with custom classification layers
- **Hyperparameter Optimization**: PSO algorithm to find optimal model parameters
- **Model Evaluation**: Comprehensive evaluation with accuracy metrics, classification reports, and confusion matrices
- **Visualization**: Training history plots, confusion matrices, and sample predictions

## Models

### TensorFlow/Keras Model

The TensorFlow implementation uses a ResNet50 base model with custom layers:
- Global Average Pooling
- Dense layer with ReLU activation
- Dropout for regularization
- Output layer with softmax activation

### PyTorch Model with PSO Optimization

The PyTorch implementation includes:
- ResNet50 base model with frozen parameters
- Custom fully connected layers
- Dropout for regularization
- Particle Swarm Optimization for hyperparameter tuning

## Results

The models achieve high accuracy in classifying US currency denominations:
- Baseline model accuracy: ~98%
- PSO-optimized model accuracy: ~99%

## Requirements

### Libraries
- Python 3.6+
- TensorFlow 2.x / Keras
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL (Pillow)
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/us-currency-classification.git
cd us-currency-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset or prepare your own dataset following the structure mentioned above.

## Usage

### Training

#### TensorFlow/Keras Model
Run the Jupyter notebook:
```bash
jupyter notebook us-currency-classification-using-deep-learning.ipynb
```

#### PyTorch Model with PSO
Run the Jupyter notebook:
```bash
jupyter notebook us-currency-classification-pytorch-pso.ipynb
```

### Inference

To classify new currency images:

1. Place your test images in the `test` directory
2. Run the inference section in either notebook
3. View the predictions with confidence scores

## Model Files

The repository includes pre-trained model files:
- `currency_classifier_baseline.pth`: Baseline PyTorch model
- `currency_classifier_pso_optimized.pth`: PSO-optimized PyTorch model
- `best_hyperparameters.json`: Best hyperparameters found by PSO

## Future Improvements

- Implement a real-time classification system using webcam input
- Develop a comprehensive GUI application with image upload functionality
- Add support for currency from other countries
- Improve model robustness to handle worn or damaged bills
- Deploy the model as a mobile application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used for this project
- The authors of ResNet50 and other pre-trained models
- The PyTorch and TensorFlow communities for their excellent documentation and support
