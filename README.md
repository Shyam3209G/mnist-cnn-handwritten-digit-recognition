# Handwritten Digit Recognition Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to recognize handwritten digits from 0 to 9. A Streamlit web application allows users to upload an image and view the model's prediction.

## Features
- Upload a handwritten digit image in PNG, JPG, or JPEG format.
- Preprocess the image for model input.
- Predict a digit from 0 to 9.
- Display the predicted digit and model confidence.
- Show the probability assigned to each digit.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Pillow
- Streamlit
- Convolutional Neural Networks (CNN)

## Project Structure
```text
mnist_cnn_app/
├── app.py
├── mnist_cnn.keras
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

4. Start the application:

```bash
python -m streamlit run app.py
```

5. Open the local URL displayed in the terminal.

## How to Use
1. Upload an image containing a handwritten digit.
2. Click **Predict Digit**.
3. View the predicted digit, confidence score, and probability for each class.

## Model Evaluation
The CNN achieved approximately **99.08% test accuracy** on the MNIST test dataset in the reported evaluation. Results on custom uploaded images may vary depending on image quality and preprocessing.

## Limitations
- The model was trained on MNIST digit images.
- Custom photographs may differ from the training images.
- Model confidence does not guarantee that a prediction is correct.

## Future Improvements
- Improve preprocessing for photographs and different handwriting styles.
- Add more informative prediction visualizations.
- Deploy the application online.

## Author
Add your name and GitHub profile here.
