from flask import Flask, render_template_string, Response, request
import os
import cv2
import pandas as pd

# Flask app setup
app = Flask(__name__)

# Paths
image_dir = 'data/Trainning_SET/Images'
pivots_csv_path = 'data/pivots.csv'

# Load pivots from CSV
def load_pivots(pivots_csv_path):
    pivots_df = pd.read_csv(pivots_csv_path)
    pivots = pivots_df[['x', 'y', 'width', 'height']].to_records(index=False)
    return [(int(row.x), int(row.y), int(row.width), int(row.height)) for row in pivots]

pivots = load_pivots(pivots_csv_path)

# Updated find_contour function
def find_contour(image_name):
    img = cv2.imread(image_name)
    if img is None:
        return None, []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=0, threshold2=100)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []

    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the approximated contour has 4 points
        if len(approx) == 4:
            # Compute the bounding box and aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Define criteria for squares (adjust thresholds as needed)
            if 0.9 < aspect_ratio < 1.1 and 20 < w < 200:
                squares.append(approx)

    return img, squares

# Function to draw contours and pivots
def draw_contours_and_pivots(image_path):
    # Calculate contours dynamically
    img, contours = find_contour(image_path)
    if img is None:
        return None

    # Draw default pivots
    for pivot in pivots:
        px, py, pw, ph = pivot
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 255, 0), 2)  # Green box for pivots

    # Draw contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for contours

    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# HTML template for rendering a single image
template = """
<!doctype html>
<html lang="en">
<head>
    <title>Single Image Viewer</title>
    <style>
        body { text-align: center; font-family: Arial, sans-serif; }
        img { max-width: 90%; margin: 10px auto; border: 1px solid #ddd; }
        form { margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Single Image Viewer</h1>
    <p>Enter the file name to calculate and render contours with pivots.</p>
    <form method="POST">
        <label for="filename">File Name:</label>
        <input type="text" id="filename" name="filename" required>
        <button type="submit">Render Image</button>
    </form>
    <hr>
    {% if img_name %}
        <div>
            <p>Image: {{ img_name }}</p>
            <img src="{{ url_for('render_image', filename=img_name) }}" alt="Rendered Image">
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    img_name = None
    if request.method == 'POST':
        img_name = request.form.get('filename')
        if img_name:
            # Check if the image exists
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                img_name = None  # Reset if file doesn't exist
    return render_template_string(template, img_name=img_name)

@app.route('/render_image/<path:filename>')
def render_image(filename):
    # Get the full image path
    image_path = os.path.join(image_dir, filename)

    img_bytes = draw_contours_and_pivots(image_path)
    if img_bytes:
        return Response(img_bytes, mimetype='image/jpeg')
    else:
        return "Image not found or could not be processed.", 404

if __name__ == '__main__':
    app.run(debug=True)
