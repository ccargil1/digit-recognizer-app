# Handwritten Digit Recognizer
A desktop application that recognizes handwritten digits using a neural network built from scratch with Python and NumPy.

The application allows users to draw a digit from 0-9 and displays the model's prediction along with its confidence score.

## Preview

## Features
- Draw handwritten digits using an interactive canvas.
- Recognize digits using a custom-trained neural network.
- Display predicted digits and confidence scores.
- Clear the canvas to make new predictions.
- Simple desktop interface built with CustonTkinter.

## How It Works
The application uses a full connected neural network trained on the MNIST handwritten digit dataset.

When a digit is drawn:
1. The drawing is captured as a grayscale image using Pillow.
2. The image is resized to 28 x 28 pixels.
3. Pixel values are normalized to the range [0, 1].
4. The processed image is passed through the neural network.
5. The predicted digit and its confidence score are displayed.

The neural network was implemented from scratch using NumPy, without TensorFlow or PyTorch.

**Neural network implementation:** [neural-network-from-scratch](https://github.com/ccargil1/neural-network-from-scratch)

## Installation
### Prerequisites
- Python 3.10 or newer
- Git

**Steps:**

**1\. Clone the repository**
```bash
git clone https://github.com/ccargil1/digit-recognizer-app.git
cd digit-recognizer-app
```
**2\. Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```
**3\. Install dependencies**
```bash
python -m pip install -r requirements.txt
```
**4\. Run the application**
```bash
python app.py
```

Make sure the trained model file is available at `models/best_model.npz`

## Technologies
- Python - Application logic
- CustomTkinter - Desktop interface
- Pillow - Drawing and image processing
- NumPy - Numerical operations and model input
- Neural Network From Scratch - Digit classification

## Related Project
[neural-network-from-scratch](https://github.com/ccargil1/neural-network-from-scratch) - The neural network implementation used by this application.