# Meets Specifications

:trophy: Terrific job with the project! I'm impressed with how you leveraged the main concepts of the Vehicle Detection lesson in your submission.

To see some ideas on using deep learning to detect vehicles, read this post on using the U-Net architecture. And for some inspiration to try combining this project and the advanced lane finding pipeline, check out this video by one of your fellow Udacity students.

## Writeup / README

### Rubric

The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

### Comments

Good work including the README and addressing each of the rubric items using the suggested writeup template.

Your report is well written, and it was cool to read about how you tackled the project. :sunglasses:

## Histogram of Oriented Gradients (HOG)

### Rubric

Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why.

### Comments

Nice job extracting the HOG features, and discussing how you arrived at your HOG parameters by experimenting with different values. You also have a nice examination of the spatially binned and color histogram features as well.

To enhance the discussion it would be nice to see some documentation of any training results you obtained with the parameter settings you experimented with. (e.g., a markdown table of settings used and prediction accuracy)

### Rubric

The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier.

### Comments

Good description of how you trained the linear SVC with the extracted HOG features and additional color features.

Suggestions:

To improve the model's accuracy, you could also try running a grid search to optimize the SVC's C parameter.
And to help reduce the feature dimensionality to speed up the pipeline, you could also consider removing the color histogram features — many students are able to exclude them and still get good results.

## Sliding Window Search

### Rubric

A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen.

### Comments

Nice work implementing the sliding window search, and describing how you arrived at your final solution of subsampling 3 scales of the HOG feature extraction.

If you're still concerned with improving the speed performance of extracting HOG features, you could also look into trying cv2.HOGDescriptor...
http://stackoverflow.com/questions/28390614/opencv-hogdescripter-python

### Rubric 

Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)

### Comments

Good work optimizing the performance of the classifier with your chosen feature vector and heatmap thresholding.

To improve the model's accuracy, other ideas you could try include:

Using the LinearSVC built-in decision_function method, which returns a confidence score based on how far a data point is from the decision boundary — higher values equate to higher confidence predictions, and you can threshold them with something like this...
if svc.decision_function(X) > threshold:
  ... add new detection
Augment the training with hard negative mining

## Video Implementation

### Rubric 

The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video.

### Comments

Terrific work processing the video and identifying the closest vehicles in the video! It appears the classifier is doing a good job of identifying cars. :sunglasses:

### Rubric 

A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur.

### Comments

Nice job filtering the vehicle detections with heatmap averaging and thresholding, and drawing the bounding boxes around the labeled maps.

## Discussion

### Rubric
Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.

### Comments
Good discussion of the issues you faced with drawing bounding boxes and the pipeline's speed, and future improvements that could be made with deep learning.

For additional ideas on performing vehicle detection, you can also check out how to use Haar Cascades.
