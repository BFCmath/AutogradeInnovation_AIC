from flask import Flask, render_template_string, Response, request
import os
import cv2
import pandas as pd
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_engineering.pivot_contour import detect_black_squares, detect_black_squares_2

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

# Function to draw contours, black squares, and pivots
def draw_all_detections(image_path):
    image_name = os.path.basename(image_path)  # Get the image name

    # Calculate contours dynamically using detect_black_squares
    img, contours = detect_black_squares(image_path)
    if img is None:
        return None, None, None

    # Calculate contours dynamically using detect_black_squares
    img_black_squares, black_squares = detect_black_squares_2(image_path)
    if img_black_squares is None:
        return None, None, None

    # Draw default pivots on both images
    for pivot in pivots:
        px, py, pw, ph = pivot
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 255, 0), 2)  # Green box for pivots
        cv2.rectangle(img_black_squares, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

    # Draw contours from detect_black_squares
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)  # Red box for contours

    # Draw contours from detect_black_squares
    for square in black_squares:
        cv2.drawContours(img_black_squares, [square], -1, (255, 0, 0), 4)  # Blue box for black squares

    # Titles for the images
    title_find_contour = f"find_contour - {image_name}"
    title_detect_black_squares = f"detect_black_squares - {image_name}"

    # Add titles to the images
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title_find_contour, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_black_squares, title_detect_black_squares, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Combine the images horizontally for display
    combined_img = cv2.hconcat([img, img_black_squares])

    _, buffer = cv2.imencode('.jpg', combined_img)
    return buffer.tobytes(), title_find_contour, title_detect_black_squares

# HTML template for rendering a single image
template = """
<!doctype html>
<html lang="en">
<head>
    <title>Single Image Viewer</title>
    <style>
        body { text-align: center; font-family: Arial, sans-serif; }
        .image-container { display: flex; justify-content: center; flex-wrap: wrap; }
        .image-wrapper { margin: 10px; text-align: center; }
        img { max-width: 90vw; border: 1px solid #ddd; }
        .title { margin-top: 5px; font-weight: bold; }
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
        <div class="image-container">
            {% if title_find_contour %}
            <div class="image-wrapper">
                <div class="title">{{ title_find_contour }}</div>
                <img src="data:image/jpeg;base64,{{ img_data }}" alt="Rendered Image">
            </div>
            {% endif %}
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    img_name, title_find_contour, title_detect_black_squares, img_data = None, None, None, None
    if request.method == 'POST':
        img_name = request.form.get('filename')
        if img_name:
            # Check if the image exists
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                img_bytes, title_find_contour, title_detect_black_squares = draw_all_detections(img_path)
                if img_bytes:
                    import base64
                    img_data = base64.b64encode(img_bytes).decode('utf-8')
            else:
                img_name = None  # Reset if file doesn't exist
    return render_template_string(template, img_name=img_name, title_find_contour=title_find_contour, title_detect_black_squares=title_detect_black_squares, img_data=img_data)

@app.route('/render_image/<path:filename>')
def render_image(filename):
    # Get the full image path
    image_path = os.path.join(image_dir, filename)

    img_bytes, _, _ = draw_all_detections(image_path)  # Ignore titles here as they are handled in the index route
    if img_bytes:
        return Response(img_bytes, mimetype='image/jpeg')
    else:
        return "Image not found or could not be processed.", 404

if __name__ == '__main__':
    app.run(debug=True)