
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢")

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit from 0 to 9.")

@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model("mnist_cnn.keras")

model = load_cnn()

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded image", width=250)

    if st.button("Predict Digit"):
        arr = np.array(image)

        # Convert to white digit on black background
        if arr.mean() > 127:
            arr = 255 - arr

        # Crop the digit from the background
        coords = np.argwhere(arr > 40)

        if len(coords) == 0:
            st.error("No digit detected. Please upload a clearer image.")
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            cropped = arr[y_min:y_max + 1, x_min:x_max + 1]
            digit = Image.fromarray(cropped.astype("uint8"))
            digit.thumbnail((20, 20))

            canvas = Image.new("L", (28, 28), 0)
            x = (28 - digit.width) // 2
            y = (28 - digit.height) // 2
            canvas.paste(digit, (x, y))

            x_input = np.array(canvas).astype("float32") / 255.0
            x_input = x_input[np.newaxis, ..., np.newaxis]

            probabilities = model.predict(x_input, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            
            confidence = float(probabilities[predicted_digit])

            st.subheader(f"Predicted digit: {predicted_digit}")
            st.write(f"Model confidence: {confidence * 100:.2f}%")

            if confidence < 0.60:
                st.warning(
                    "The model is uncertain about this image. "
                    "Try a clearer, centered handwritten digit."
                )
            else:
                st.success(
                    "The model is reasonably confident in this prediction."
                )

            