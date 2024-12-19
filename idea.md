# IDEA

## TODO

+ [X] Load data
+ [X] Observe data
+ [X] Find current solution in public (not dive deep yet)
+ [X] Save the training set into a csv
+ [ ]  Process the error in the labels
  + [X] Detect the error
  + [X] Make a web reader to analyze the error
  + [X] Predict and confirm the error
  + [ ] Process the error
    + [X] Remove the duplicates
    + [ ] Remove the redundants
    + [ ] Replace the missings
  + [ ] Cutting images
+ [ ] Dive deep into public solution
+ [ ] Propose solutions
+ [ ] Build pipeline
+ [ ] Try opencv approach (augmentation)

## Observations

### First observation on the dataset

+ There are 5200 images in first trainging batch
+ Training set will give us:
  + raw images
  + labeled images
  + yolo labels
+ Testing set will only give us raw images
+ There are some repetitive error in the training labels (that we need to process ourselves)
+ The images are not even (probably due to scanning process)
+ Duplicate choices in the same question
+ No choice in some question

### Error in the labels

+ Missing some bounding boxes
+ Duplicate bounding boxes
+ Redundant bounding boxes
+ Bounding boxes is not even

### Second observation on the dataset

+ It seems that all the images have the same format
+ The trainging data is split into each batch of 260 iters
+ Within the same batch, the images have the same type of error

### Third observation on the dataset

+ The bounding boxes are not even
+ We can using the black squares to get relative position of each bounding box window.
+ There are yellow streaks in some images

### Cutting images

+ Based on the black squared contours at the corners, we can effectively detect them and cutting the images.
+ But there are still some error from the images so we cannot rely completely on the contour detective algorithm.
+ So we will use some default pivot to check if the algorithm detecting the contour in those regions.
