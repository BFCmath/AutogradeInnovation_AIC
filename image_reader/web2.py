from flask import Flask, render_template_string, request, Response
import pandas as pd
import os
import cv2

app = Flask(__name__)

# Directories for images and labels
image_dir = 'data/Trainning_SET/Images'
label_dir = 'data/Trainning_SET/Labels'
# label_dir = 'data/created/processed_label'

# Load the CSV file and get the image paths
# df = pd.read_csv('data/created/label_3120_3380.csv')
df = pd.read_csv('data/created/training.csv')
image_paths = [os.path.join(image_dir, path) for path in df['image_name'].tolist()]

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
        if len(parts) < 5:
            continue  # Skip invalid lines
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
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if mode == "score":
            label = f"{class_id}"
        elif mode == "count":
            label = f"{i}"
        else:
            label = ""

        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(img, label, (text_x, text_y), font, font_scale, color2, text_thickness)
    
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# Function to generate specific indices following the pattern: 1, 259, 260, 519, 520, ..., up to 5200
def generate_specific_indices(max_id=5200):
    indices = []
    current = 1
    step_large = 258  # Difference between 1 and 259 is 258
    step_small = 1    # Difference between 259 and 260 is 1
    toggle = True

    while current <= max_id:
        index = current - 1  # Convert to zero-based index
        if index < len(image_paths):
            indices.append(index)
        else:
            break  # Stop if index exceeds available images
        if toggle:
            current += step_large
        else:
            current += step_small
        toggle = not toggle

    return indices

# HTML template with Resizable Split Panes
template = """
<!doctype html>
<html lang="en">
<head>
    <title>Image Viewer with Bounding Boxes</title>
    <style>
        body { 
            text-align: center; 
            font-family: Arial, sans-serif; 
            margin: 0; 
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background-color: #f4f4f4;
            padding: 10px 0;
        }
        .split-container {
            display: flex;
            flex: 1;
            height: calc(100vh - 70px); /* Adjust based on header height */
        }
        .pane {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            box-sizing: border-box;
        }
        .pane img {
            max-width: 100%;
            height: auto;
            display: block;
            margin-bottom: 20px;
            border: 1px solid #ddd;
        }
        .splitter {
            width: 5px;
            background-color: #ccc;
            cursor: ew-resize;
        }
    </style>
</head>
<body>
    <header>
        <h1>Image Viewer with YOLO Bounding Boxes</h1>
    </header>
    <div class="split-container">
        <div class="pane" id="pane1">
            <h2>Mode: Score</h2>
            {% for img_path in images %}
                <img src="{{ url_for('yolo_image', filename=img_path) }}?mode=score" alt="Mode Score">
            {% endfor %}
        </div>
        <div class="splitter" id="splitter"></div>
        <div class="pane" id="pane2">
            <h2>Mode: Count</h2>
            {% for img_path in images %}
                <img src="{{ url_for('yolo_image', filename=img_path) }}?mode=count" alt="Mode Count">
            {% endfor %}
        </div>
    </div>

    <script>
        // Resizable Split Panes Implementation
        const splitter = document.getElementById('splitter');
        const pane1 = document.getElementById('pane1');
        const pane2 = document.getElementById('pane2');

        splitter.addEventListener('mousedown', function(e) {
            e.preventDefault();
            document.addEventListener('mousemove', resize);
            document.addEventListener('mouseup', stopResize);
        });

        function resize(e) {
            const containerOffsetLeft = splitter.parentElement.offsetLeft;
            const pointerRelativeXpos = e.clientX - containerOffsetLeft;
            const pane1Width = pointerRelativeXpos;
            const containerWidth = splitter.parentElement.clientWidth;
            const minWidth = 100; // Minimum width in px

            if (pane1Width > minWidth && (containerWidth - pane1Width) > minWidth) {
                pane1.style.width = pane1Width + 'px';
                pane2.style.width = (containerWidth - pane1Width - splitter.offsetWidth) + 'px';
            }
        }

        function stopResize() {
            document.removeEventListener('mousemove', resize);
            document.removeEventListener('mouseup', stopResize);
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    # Generate the specific list of indices
    specific_indices = generate_specific_indices(max_id=5200)
    
    # Select images based on the generated indices
    images = [image_paths[i] for i in specific_indices if i < len(image_paths)]
    
    return render_template_string(template, images=images)

@app.route('/yolo_image/<path:filename>')
def yolo_image(filename):
    mode = request.args.get('mode', 'score')  # Get mode from query parameter
    # Extract image name from the path and find the corresponding label file
    image_name = os.path.basename(filename)
    label_path = os.path.join(label_dir, f"{os.path.splitext(image_name)[0]}.txt")
    img_bytes = draw_yolo_bboxes(filename, label_path, mode)
    if img_bytes:
        return Response(img_bytes, mimetype='image/jpeg')
    else:
        return "Image or labels not found.", 404

if __name__ == '__main__':
    app.run(debug=True)
