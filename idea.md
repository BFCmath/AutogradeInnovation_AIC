# IDEA

## TODO

- [x] **Load data**  
- [x] **Observe data**  
- [x] **Find a public solution** (initially without in-depth exploration)  
- [x] **Save the training set to a CSV**  
- [x] **Examine the public solution more closely**  
- [x] **Propose potential solutions**  
- [x] **Address labeling errors**  
  - [x] **Identify errors**  
  - [x] **Create a web reader to assess errors**  
  - [x] **Predict and confirm identified errors**  
  - [x] **Crop images**  
    - [x] Detect contours  
    - [x] Sort contours  
  - [x] **Correct errors**  
    - [x] Remove duplicates  
    - [x] Remove redundant labels  
    - [x] Replace missing labels  
  - [x] **Replace 2-labeled bounding boxes with correct labels**  
- [x] **Build the pipeline**  
  - [x] Contour detection  
  - [x] Contour sorting  
  - [x] Image cropping  
  - [x] End-to-end pipeline creation  
- [ ] **Fine-tune the model**  
  - [ ] Data augmentation  
  - [x] Use YOLOv8  
  - [ ] Error analysis  
- [x] **Predict on Test Set 1**  
  - [x] Contour detection  
  - [x] Contour sorting  
  - [x] Image cropping  
  - [x] Model inference  

---

## Observations

### First Observation of the Dataset

- There are 5,200 images in the first training batch.  
- The training set provides:  
  - Raw images  
  - Labeled images  
  - YOLO labels  
- The testing set only contains raw images.  
- There are recurring label errors that need manual correction.  
- Images vary in size and orientation, possibly due to the scanning process.  
- Some questions contain duplicate choices.  
- Some questions have no choices at all.

### Labeling Errors

- Missing bounding boxes  
- Duplicate bounding boxes  
- Redundant bounding boxes  
- Bounding boxes that are not aligned properly

### Second Observation of the Dataset

- All images follow a consistent format.  
- The training data is grouped into batches of 260 iterations each.  
- Within a single batch, images tend to exhibit the same type of error.

### Third Observation of the Dataset

- Bounding boxes are often misaligned.  
- Black squares can be used as reference points for bounding box positions.  
- Some images have yellow streaks.

### Cutting Images

- Based on the black square contours in each corner, we can detect and crop images effectively.  
- Due to irregularities in some images, contour detection alone is not fully reliable.  
- We will use a default pivot approach to verify whether the algorithm detects contours in specific regions.  
- If some contours remain undetected, we will try a stacking approach to detect those contours (without relying on the default pivot).  
- If contours are still missing, we will apply manual detection using predefined rules.  
- After confirming all contours, we can finally crop the images according to our rules.  
- We must also convert bubble positions into YOLOv8 format for each cropped image.

### Processing Errors

- **Duplicate Bounding Boxes**:  
  - Use IOU (Intersection Over Union) to identify these erroneous bounding boxes.
- **Redundant Bounding Boxes**:  
  - Visualize and apply rules to remove unnecessary bounding boxes.
- **Missing Bounding Boxes**:  
  - Visualize and replace these bounding boxes based on predefined rules.

### Fourth Observation of the Dataset

- Some duplicate bubbles have conflicting classes.

### Handling 2-Class Bubbles

- During the cropping process, bounding boxes with two classes are initially labeled as “2.”  
- We then replace these 2-class labels with the correct label using the following method:  
  1. **Whole Bounding Box Check**: Examine the entire bounding box and measure the ratio of black pixels.  
  2. **Half Bounding Box Check**: Examine half of the bounding box and measure the ratio of black pixels.  
  3. **Thresholds**: Use two separate thresholds (one for the entire bounding box and one for the half).  
     - If both ratios exceed their respective thresholds, the final label is “0” (filled).  
     - Otherwise, the final label is “1” (unfilled).

---
