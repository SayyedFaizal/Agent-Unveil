from flask import Flask, render_template, request
import cv2
import numpy as np
import pytesseract
from transformers import pipeline
import spacy

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/process', methods = ['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded.', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file.', 400
    if file:
        print("Received file:",file.filename)
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("grey.png", grey)
            text = pytesseract.image_to_string(grey)
            print(text)
            summarizer = pipeline("summarization",model="sshleifer/distilbart-cnn-12-6")
            summary = summarizer(text,max_length=69)
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            product_name = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]
            urls = [ent.text for ent in doc if ent.like_url]
            output_text = f'{summary[0]["summary_text"]} \n {product_name} \n {urls}'
            return render_template('index.html', extracted_text=output_text)
        except Exception as e:
            return f"An error occured: {e}", 500