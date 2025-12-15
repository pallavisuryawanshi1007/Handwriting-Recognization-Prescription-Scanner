
import os 
import cv2
import numpy as np
import easyocr
import pandas as pd
from flask import Flask, render_template, request
from fuzzywuzzy import process
import webbrowser

# Initialize Flask App
app = Flask(__name__)

# Load Medicine Dataset
dataset_path = "dataset/Training/training_labels.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")

df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip().str.upper()
medicine_list = df["MEDICINE_NAME"].dropna().unique().tolist()

# Initialize OCR Reader
reader = easyocr.Reader(['en'])

# Ensure Uploads Folder Exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def preprocess_image(image_path):
    """ Preprocess the image for better OCR results """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    processed_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    return processed_image


def extract_text(image_path):
    """ Extract text using EasyOCR """
    result = reader.readtext(image_path, detail=1)  # Keeping details for accuracy scores
    extracted_words = [(text, score) for (bbox, text, score) in result]
    return extracted_words


def match_medicine(extracted_words):
    """ Match extracted words to known medicine names using fuzzy matching """
    detected_medicines = []

    for word, score in extracted_words:
        match_result = process.extractOne(word, medicine_list, score_cutoff=70)
        if match_result:
            best_match, match_score = match_result
            detected_medicines.append((best_match, match_score))

    return list(set(detected_medicines))  # Remove duplicates


@app.route("/")
def home():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Prescription OCR</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
        }

        .container {
            text-align: center;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            width: 350px;
            animation: fadeIn 1.5s ease-in-out;
        }

        h2 {
            font-weight: 600;
            color: #333;
            margin-bottom: 20px;
        }

        .file-input {
            display: none;
        }

        .custom-file-upload {
            display: inline-block;
            background: #2575fc;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: 0.3s;
        }

        .custom-file-upload:hover {
            background: #1a5fd6;
        }

        #file-name {
            margin-top: 10px;
            font-size: 14px;
            color: #555;
        }

        .upload-btn {
            background: #28a745;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            transition: 0.3s;
        }

        .upload-btn:hover {
            background: #218838;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Upload Prescription Image</h2>
        <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
            <label for="file-upload" class="custom-file-upload">Choose Image</label>
            <input id="file-upload" class="file-input" type="file" name="file" accept="image/*" required>
            <p id="file-name">No file chosen</p>
            <button type="submit" class="upload-btn">Upload</button>
        </form>
    </div>

    <script>
        document.getElementById("file-upload").addEventListener("change", function() {
            document.getElementById("file-name").textContent = this.files[0] ? this.files[0].name : "No file chosen";
        });
    </script>
</body>
</html>
    """


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "<h3 style='color:red;'>No file uploaded!</h3>"

    file = request.files["file"]
    if file.filename == "":
        return "<h3 style='color:red;'>No selected file!</h3>"

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    print("📸 Image Uploaded:", file_path)

    # Process Image
    extracted_words = extract_text(file_path)
    detected_medicines = match_medicine(extracted_words)

    if not detected_medicines:
        return """
        <div style='text-align:center;'>
            <h2 style='color:red;'>No Medicines Detected</h2>
            <a href='/' style='text-decoration:none; padding:10px 15px; background:#28a745; color:white; border-radius:5px;'>🔙 Go Back</a>
        </div>
        """

    # Generate buttons with accuracy and search links
    medicine_buttons = ""
    for medicine, accuracy in detected_medicines:
        search_url = f"https://www.google.com/search?q={medicine}+medicine+details"
        medicine_buttons += f"""
            <button onclick="window.open('{search_url}', '_blank')" 
                    style="display:block; width:80%; margin:10px auto; padding:10px; background:#2575fc; 
                           color:white; border:none; border-radius:5px; cursor:pointer; font-size:16px;">
                {medicine} (Accuracy: {accuracy}%)
            </button>
        """


        

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detected Medicines</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {{
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #ff9a9e, #fad0c4);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
        }}

        .container {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            text-align: center;
            width: 350px;
            animation: fadeIn 1.5s ease-in-out;
        }}

        h2 {{
            color: #4CAF50;
            font-weight: 600;
            margin-bottom: 20px;
        }}

        .medicine-button {{
            background: #2575fc;
            color: white;
            border: none;
            padding: 12px;
            width: 100%;
            margin: 8px 0;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s, transform 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
        }}

        .medicine-button:hover {{
            background: #1a5fd6;
            transform: scale(1.05);
        }}

        .medicine-name {{
            flex: 1;
            text-align: left;
        }}

        .accuracy {{
            font-size: 14px;
            color: #f1f1f1;
            font-weight: bold;
        }}

        .back-btn {{
            display: inline-block;
            margin-top: 15px;
            padding: 10px 20px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: 0.3s;
        }}

        .back-btn:hover {{
            background: #218838;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Detected Medicines</h2>
        <div>
            {''.join([
                f'<button class="medicine-button" onclick="searchMedicine(\'{med}\')">'
                f'<span class="medicine-name">{med}</span> '
                f'<span class="accuracy">{acc}%</span></button>'
                for med, acc in detected_medicines
            ])}
        </div>
        <a href="/" class="back-btn">🔙 Go Back</a>
    </div>

    <script>
        function searchMedicine(medicine) {{
            window.open(`https://www.google.com/search?q=${{medicine}}+medicine+details`, '_blank');
        }}
    </script>
</body>
</html>
"""



if __name__ == "__main__":
    app.run(debug=True)


