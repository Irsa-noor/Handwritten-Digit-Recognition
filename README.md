# Handwritten Digit Recognition with Neural Networks

A machine learning project that recognizes handwritten digits (0-9) using a deep neural network trained on the MNIST dataset. Features an interactive web interface built with Streamlit.

##  Project Overview

**What does it do?**
- Train a neural network to recognize handwritten digits
- Draw a digit in a web app → AI predicts what it is
- Achieves ~98% accuracy on test data

**Technologies Used:**
- **Machine Learning**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Data**: MNIST (70,000+ handwritten digit images)
- **Language**: Python 3.8+

##  Model Architecture
Input Layer (784 neurons)

↓

Dense Layer 1 (128 neurons, ReLU activation)

↓

Dropout (20% regularization)

↓

Dense Layer 2 (64 neurons, ReLU activation)

↓

Dropout (20% regularization)

↓

Dense Layer 3 (32 neurons, ReLU activation)

↓

Output Layer (10 neurons, Softmax activation)

↓

Predictions (digits 0-9)

**Why this architecture?**
- Multiple layers: Learns complex patterns
- Dropout: Prevents overfitting
- ReLU activation: Helps learn non-linear relationships
- Softmax output: Probability distribution over 10 classes

##  Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.2% |
| Test Accuracy | 98.1% |
| Training Samples | 60,000 |
| Test Samples | 10,000 |
| Image Size | 28×28 pixels |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup

**1. Clone or download this repository**
```bash
git clone https://github.com/yourusername/handwritten-digit-recognition
cd handwritten-digit-recognition
```

**2. Create virtual environment**
```bash
python -m venv venv
```

**3. Activate virtual environment**

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Project

**Step 1: Train the model**
```bash
python train_model.py
```
 This takes 2-3 minutes. It will:
- Download MNIST dataset (12MB)
- Train the neural network
- Save the trained model to `models/digit_recognition_model.h5`

**Step 2: Launch the web app**
```bash
streamlit run app.py
```

**Step 3: Use the app**
- Open your browser (automatically opens at http://localhost:8501)
- Draw a digit in the canvas
- Watch the AI predict in real-time!

## Project Structure
handwritten-digit-recognition/

│

├── train_model.py              # Training script

├── app.py                       # Streamlit web app

├── requirements.txt            # Python dependencies

├── README.md                   # This file

│

└── models/                     # (Created after running train_model.py)

├── digit_recognition_model.h5

└── training_history.pkl

## 🔍 How It Works

### Training Phase
1. **Load Data**: Download MNIST dataset (70,000 images)
2. **Preprocess**: 
   - Normalize pixel values (0-255 → 0-1)
   - Flatten 28×28 images to 784-dimensional vectors
3. **Build Model**: Create neural network with 4 layers
4. **Train**: Use 60,000 images to teach the model
5. **Evaluate**: Test on 10,000 unseen images
6. **Save**: Store trained model for later use

### Prediction Phase
1. User draws digit in web app
2. Image is captured and preprocessed
3. Converted to 28×28 pixels
4. Inverted colors (white background, black drawing)
5. Flattened to 784 dimensions
6. Fed to trained neural network
7. Output: Probability for each digit (0-9)
8. Display: Predicted digit with confidence %

##  Key Concepts Explained

### Neural Network
An AI model inspired by the brain. It learns by adjusting weights through training.

### Dropout
Randomly "turns off" some neurons during training to prevent overfitting (memorizing instead of learning).

### Epochs
One complete pass through all training data. 15 epochs = 15 passes through 60,000 images.

### Softmax Activation
Converts raw outputs into probabilities that sum to 100%.

### Accuracy
Percentage of correct predictions on test data.

## 🎓 Learning Outcomes

By completing this project, you'll understand:
- How neural networks work
- Data preprocessing for ML
- Model training and evaluation
- Building interactive web apps with Streamlit
- Real-world ML pipeline from data to deployment

##  Troubleshooting

### "Model not found" error
**Solution:** Run `python train_model.py` first

### "ModuleNotFoundError" (e.g., tensorflow not found)
**Solution:** Make sure virtual environment is activated and run `pip install -r requirements.txt`

### Slow performance
**Solution:** TensorFlow first run downloads large files. Be patient on first execution.

### Drawing doesn't work
**Solution:** Make sure you have streamlit-drawable-canvas installed: `pip install streamlit-drawable-canvas`

## Improvements & Next Steps

**To enhance this project:**
1. Add data augmentation for better accuracy
2. Use convolutional neural networks (CNN) for better performance
3. Save training history graphs
4. Deploy to cloud (Heroku, AWS, Google Cloud)
5. Add batch prediction for images
6. Fine-tune hyperparameters

##  Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Sequential API](https://keras.io/guides/sequential_model/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Neural Networks Explained](https://www.3blue1brown.com/lessons/neural-networks)

