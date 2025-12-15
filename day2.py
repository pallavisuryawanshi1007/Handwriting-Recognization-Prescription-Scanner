import cv2
import numpy as np
import easyocr
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from Levenshtein import ratio
import os
import webbrowser

# Load Dataset for Medicine Recognition
dataset_path = r"dataset/Training/training_labels.csv"
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip().str.upper()
medicine_list = df["MEDICINE_NAME"].dropna().unique().tolist()

# Initialize OCR Reader
reader = easyocr.Reader(['en'])

def upload_image():
    """ Opens file dialog to select an image """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

def preprocess_image(image_path):
    """ Loads & enhances image for OCR """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    processed_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    return processed_image

def extract_text(image_path):
    """ Extracts text using EasyOCR """
    result = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(result)
    print("\n🔹 Extracted Text:", extracted_text)
    return extracted_text.split()

def match_medicine(extracted_words):
    """ Matches extracted words to closest medicine names """
    detected_medicines = []
    for word in extracted_words:
        best_match = max(medicine_list, key=lambda med: ratio(word.lower(), med.lower()))
        best_score = ratio(word.lower(), best_match.lower())
        
        print(f"🔍 Matching {word} → {best_match} (Score: {best_score:.2f})")
        if best_score > 0.75:
            detected_medicines.append(best_match)
    
    return detected_medicines[:len(extracted_words)]

def get_medicine_info(medicine_name):
    """ Open Google search for medicine details """
    search_url = f"https://www.google.com/search?q={medicine_name}+medicine+details"
    print(f"No information found. Opening Google search: {search_url}")
    webbrowser.open(search_url)

# 🔹 Run
image_path = upload_image()
if not image_path:
    print("\n❌ No Image Selected. Exiting.")
    exit()

processed_image = preprocess_image(image_path)
extracted_words = extract_text(image_path)
detected_medicines = match_medicine(extracted_words)

print("\n🎯 Final Medicines:", ", ".join(detected_medicines))

for medicine in detected_medicines:
    get_medicine_info(medicine)
