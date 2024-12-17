from flask import Flask, render_template_string, request, send_from_directory
import pandas as pd
import os
import cv2
from flask import Response

app = Flask(__name__)

# Load the CSV file and get the image paths
df = pd.read_csv('data\created\\training.csv')
image_paths = [os.path.join('data/Trainning_SET/Images', path) for path in df['image_name'].tolist()]

# Function to draw YOLO bounding boxes
def draw_yolo_bboxes(image_path, label_path, mode="score"):
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width, _ = img.shape

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return img

    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height

        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_height / 2)

        color = (0, 255, 0)
        color2 = (255, 0, 0)
        thickness = 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if mode == "score":
            label = f"{class_id}"
        elif mode == "count":
            label = f"{i}"
        else:
            label = ""

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 2
        cv2.putText(img, label, (x1+(i%2)*10, y1 - 5+(i%3)*7), font, font_scale, color2, text_thickness)
    
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# HTML template for displaying images and input box
template = """
<!doctype html>
<html lang="en">
<head>
    <title>Image Viewer with Bounding Boxes</title>
    <style>
        body { text-align: center; font-family: Arial, sans-serif; }
        input { margin: 0px; padding: 0px; }
        img { max-width: 90%; display: inline-block; margin: 0px; border: 1px solid #ddd; }
        .image-container { display: flex; justify-content: center; gap: 0px; }
    </style>
</head>
<body>
    <h1>Image Viewer with YOLO Bounding Boxes</h1>
    <form method="POST">
        <label for="start">Start Index:</label>
        <input type="number" id="start" name="start" min="0" required>
        <label for="end">End Index:</label>
        <input type="number" id="end" name="end" min="0" required>
        <button type="submit">View Images</button>
    </form>
    <hr>
    {% if images %}
        {% for img_path in images %}
        <div class="image-container">
            <div>
                <p>Mode: Score</p>
                <img src="{{ url_for('yolo_image', filename=img_path, mode='score') }}" alt="Mode Score">
            </div>
            <div>
                <p>Mode: Count</p>
                <img src="{{ url_for('yolo_image', filename=img_path, mode='count') }}" alt="Mode Count">
            </div>
        </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    images = []
    if request.method == 'POST':
        try:
            start = int(request.form.get('start'))
            end = int(request.form.get('end'))
            if start >= 0 and end >= start:
                images = image_paths[start:end + 1]
            else:
                images = []
        except Exception as e:
            print(e)
            images = []
    return render_template_string(template, images=images)

@app.route('/yolo_image/<path:filename>/<mode>')
def yolo_image(filename, mode):
    image_path = filename
    label_path = filename.replace(".jpg", ".txt").replace("Images", "Labels")
    img_bytes = draw_yolo_bboxes(image_path, label_path, mode)
    if img_bytes:
        return Response(img_bytes, mimetype='image/jpeg')
    else:
        return "Image or labels not found.", 404

if __name__ == '__main__':
    app.run(debug=True)
