from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import os

app = Flask(__name__)

# Load your YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/your/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img)
        results.render()  # render the bounding boxes on the image

        # Save the image with bounding boxes to the static directory
        output_path = os.path.join('static', 'output.jpg')
        results.imgs[0].save(output_path)

        return jsonify({'result': 'success', 'output_path': output_path})

if __name__ == '__main__':
    app.run(debug=True)
