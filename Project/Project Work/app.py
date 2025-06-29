from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model_path = "CDX_Best_RestNet50.h5"
model = load_model(model_path)


def preprocess_image(img):
    try:
        img = img.convert("RGB")  # Convert to RGB in case the image is not in RGB format
        img = img.resize((224, 224))  # Resize image to match model's expected sizing
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        print("Image preprocessed successfully.")
        return img_array
    except Exception as e:
        print("Error during image preprocessing:", str(e))
        return None


def predict(image):
    try:
        processed_img = preprocess_image(image)
        if processed_img is not None:
            predictions = model.predict(processed_img)
            covid_probability = predictions[0][0]
            print("Predictions:", predictions)
            if covid_probability > 0.5:  # Adjust threshold as needed
                return "COVID Positive"
            else:
                return "COVID Negative"
        else:
            return "Error occurred during image preprocessing"
    except Exception as e:
        print("Error during prediction:", str(e))
        return "Error occurred while processing the image"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    result = None
    if request.method == "POST":
        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", result="No selected file")

        try:
            img = Image.open(io.BytesIO(file.read()))
            img.verify()  # Verify if the file is an image
            result = predict(img)
        except Exception as e:
            print("Error during file upload:", str(e))
            return render_template(
                "index.html", result="Invalid image file or unreadable image"
            )

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
