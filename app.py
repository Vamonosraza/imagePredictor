from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_bootstrap import Bootstrap
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

app = Flask(__name__)
# style the app with bootstrap
Bootstrap(app)

# load the CLIP model and processor for inference
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# custom categories for the model
categories = ["Bright yellow blooms", "towering purple spines","Red spiky "
                                                               "stem",
              "delicate blue petals", "bright white petals"]

# path to dataset
dataset_path = "./images"


@app.route('/')
# renders the index.html template
def index():
    return render_template('index.html')

@app.route('/images')
# returns a list of image paths in json format
def images():
    image_paths = [os.path.join('images', img) for img in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, img))][:5]
    return jsonify(image_paths)

@app.route('/images/<path:filename>')
# returns an image file from the images directory (dataset)
def send_image(filename):
    return send_from_directory(dataset_path, filename)

@app.route('/upload', methods=['POST'])
# handling the image uploads, processing the image with the CLIP model and
# returns the classification results in json format
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        custom_category = request.form.get('customCategory', '').strip()
        all_categories = categories +[custom_category] if custom_category else categories

        image = Image.open(file.stream)
        inputs = processor(text=all_categories, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        results = {category: prob.item() for category, prob in zip(all_categories, probs[0])}
        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)