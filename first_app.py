from flask import Flask, render_template, request
import os
import cv2
import numpy as np
# import pytesseract
import easyocr
# from transformers import pipeline
import spacy
# from keybert import KeyBERT
reader = easyocr.Reader(['en'], gpu=False)
nlp = spacy.load("en_core_web_sm")

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
            result = reader.readtext(grey)
            text = " ".join([res[1] for res in result])
            doc = nlp(text)
            urls = [ent.text for ent in doc if ent.like_url]
            # text = pytesseract.image_to_string(grey)
            # summarizer = pipeline("summarization",model="sshleifer/distilbart-cnn-12-6")
            # summary = summarizer(text,max_length=69)
            # nlp = spacy.load("en_core_web_sm")
            # doc = nlp(text)
            # kw_model = KeyBERT()
            # keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5)
            # urls = [ent.text for ent in doc if ent.like_url]
            # keys = ""
            # for k in keywords:
            #     keys += k[0]+"<br>"
            # output_text = f'{text} <br><b>Extracted Urls present in the image - </b>{urls} <br><b>Keywords to Search:</b><br>{keys}'
            output_text = f'{text} <br><b>Extracted Urls present in the image - </b>{urls}'
            return render_template('index.html', extracted_text=output_text)
        except Exception as e:
            return f"An error occured: {e}", 500
        

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render dynamic port
    app.run(host="0.0.0.0", port=port)