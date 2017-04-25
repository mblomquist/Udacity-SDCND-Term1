# Udacity SDCND: Vehicle Detection and Tracking - Project 5
Udacity Self-Driving Car Nanodegree Project 5 - Vehicle Detection and Tracking

## Project Goals:
The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
- Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

## Project Code

The project was completed in a Jupyter Notebook and the sections of the notebook have the same section names as provided below. The project notebook can be found at this url:
https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/Project5.ipynb

## Feature Extraction

I chose to integrate three methods of feature extraction to create a robust classifier that performed well on testing images as well as during the video pipeline for the project. Those methods were Histogram of Oriented Gradients, Spatial Feature Extraction, and Histogram Color Feature Extraction. Each of the methods is described in detail below and pictures of the accompanying outputs have been provided. The test images used in the method below can be seen here, unaltered for reference.

![test_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/test_img.png?raw=true)

### Histogram of Oriented Gradients (HOG)

The Histogram of Oriented Gradients (HOG) feature extraction method was presented in the lessons as well as referenced in a number of research publications [1, 2], so I treaded this method as a good starting point. I explored a number of color spaces (including RGB, HSV, and YCrCb) in conjunction with HOG feature extraction and chose to implement the YCrCb color space as it provided the most visible variation between vehicle and non-vehicle images. For each of the YCrCb color space, I extracted each channels (i.e. Y, Cr, and Cb) and put the extracted channel through the HOG function from sci-kit image. The results for a typical vehicle image and non-vehicle image are shown below.

![hog_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/hog_img.png?raw=true)

The parameters chosen for the hog function did not vary from the parameters presented in the classroom lessons. After a small amount of experimentation, the classifier results didn’t significantly improve and the duration of image processing pipeline increased by a factor of two. For those reasons, these values were maintained:
```
pix_per_cell = (8,8)
cells_per_block = (2,2) 
orient = 9
```

### Spatial Feature Extraction

The Spatial Feature Extraction method was also presented in the lessons and provided approximately 3% improvement in the test accuracy when testing the support vector classifier. Therefore, I implemented this method into the pipeline for feature extraction. The spatial function for feature extraction takes in an image and breaks it into three component color channels. Each channel is then resized to 32x32 pixels and the compressed into a single dimension array. The three channels are then stacked together into a long feature array and returned. The function is provided below and corresponds to what was presented in the lessons.

```
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
```

The resulting features for a typical vehicle image and a non-vehicle image are shown below.

![spatial_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/spatial_img.png?raw=true)

### Histogram Color Feature Extraction

The Histogram Color Feature Extraction method was presented in the lesson and improved the performance of the classifier when used in the video pipeline. For this reason, I included the method in the final, submitted pipeline.  Similar to the Spatial Feature Extraction function, the Histogram Color function extracts each color channel from the input image. Next, a histogram is taken over the single-channel image with 32 bins and the histogram data for each channel is stacked and returned. The function is provided below and corresponds to what was presented in the lessons.

```
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
```

The resulting features for a typical vehicle image and a non-vehicle image are shown below.

![hist_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/hist_img.png?raw=true)

## Support Vector Classifier

Once all of the features were extracted from the vehicle and non-vehicle images using the HOG, spatial, and histogram color extraction methods, I created a concatenated array with all of the features. Next, I normalized all of the features using the ```StandardScaler``` function from ```sklearn.preprocessing```. The StandardScaler function removes the mean of the feature array and scales the data to a unit variance. This creates a Gaussian-like distribution with a mean of zero and a unit variance. Additionally, it normalizes the feature arrays that may have different magnitudes, ensuring different feature extraction methods are weighted equally. 

After normalizing the data, I shuffle the data to prevent the classifier from over-fitting the data based on the data sequence. Once shuffled, I split the test data into a set of training data and a set of test data with a ratio of 80/20 for training and testing data, respectively. A linear, support-vector classifier is then fit with the training data and tested against the test data. For the submitted classifier, I achieved an accuracy of 99.53%. 

I chose to utilize a support-vector classifier has it seemed appropriate from previous research [1,2] and I had little experience using SVMs (support-vector machines). I was able to achieve a high accuracy when combining the SVM classifier with the 3 methods of feature extraction, so I continued using it. Additionally, the SVM classifier preformed quickly when doing prototype testing which added value to its use in the video pipeline.

## Sliding Window Search

I implemented a sliding window search in a mostly identical way to that shown in the lesson. First, I would scale the input image data to values between 0 and 1 for .jpg images. The SVC was trained using .png images which have values between 0 and 1, so scaling is required for anything different.

Next, I restricted the sliding window area between y-values of 400 and 656-pixels to cover only the areas that cars appear. I’ve provided an image below to clarify the viewing area.

![crop_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/crop_img.png?raw=true)

I chose to implement a scaling factor in the sliding window function to easily modify the search window size for any given image. From experimentation, I ended up choosing to use scaling factors of 0.75, 1.5, and 2.0 which correspond to window sizes of 40x40, 96x96, and 128x128-pixels. Additionally, I chose to overlap the windows by 75% to generate a thorough scan of the image. In retrospect, reducing the overlap may have significantly improved the performance (duration) of the image processing pipeline. For future work, this aspect should be tested to validate the minimum effective overlap.

![scale_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/scale_img.png?raw=true)

## Image Processing Pipeline

I created an image processing pipeline by creating a function ```find_cars``` that sequences all of the previously mentioned steps. First, the scaling factor is calcualated to generate the search window sizes. Next, I chose to implement the HOG feature extraction on the entire image prior to performing the window search to improve the efficiency of the pipeline. This reduced the need to perform the HOG extraction on each of the sub-sampled windows, which significantly improved the performance of the pipeline. 

After extracting the HOG features for the entire image, the sliding window search was performed and the spatial and histogram color feature extraction methods were performed on the sub-sampled images as this ended up providing a methodology for feature extraction similar to that performed when generating data to train the classifier. 

All of the feature arrays were stacked together and the previously trained SVC was used to predict the existence of a car in the particular sliding window image. If the prediction resulted in a car, a bounding box was created for that particular search window.

The ```find_cars``` function is provided below for reference.
```
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    
    # Create a Copy Image to Draw on
    draw_img = np.copy(img)
    
    # Test Images are in .jpg format so scaling is required.
    img = img.astype(np.float32)/255
    
    # Constrain the Searchable Image Area to ystart and ystop
    img_tosearch = img[ystart:ystop,:,:]
    
    # Convert to YCrCb Color Space
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    
    # If Scaling is Applied, Reshape the Image
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # Extract Each Color Channel
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define Window Step Blocks 
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Search Subspaces of the image for Cars
    for xb in range(nxsteps):
        for yb in range(nysteps):
            
            # Find Current Step
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # Define Stepping Area
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg)
            hist_features = color_hist(subimg)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            # Draw Bounding Box if Prediction is True
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img
``` 

In order to demonstrate the simple pipeline, I’ve provided a number of test images below from a sequence of road.

![test_img01](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/test_box01.png?raw=true)
![test_img02](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/test_box02.png?raw=true)
![test_img03](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/test_box03.png?raw=true)
![test_img04](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/test_box04.png?raw=true)
![test_img05](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/test_box05.png?raw=true)

## Image Processing Pipeline Implementation

As seen in the above images, the previously noted image processing pipeline performs fairly well on sequences of images, but generates false positives and doesn’t adequately cover the entire surface of the white car in multiple frames. To correct these issues, I’ve implemented a filtering process using a heatmap and the skimage function labels. The details of those processes are mentioned below.

## Heatmap Filtering

Heatmap filtering was implemented in the video processing pipeline to filter false positives resulting from error in the classifier. To achieve this, a function ```add_heat``` was created to add values to pixels of a blank image where bounding boxes are found. This process is repeated for each bounding box found via the find_cars function and the result is an image the generates hot-spots for pixels contained within multiple bounding boxes. 

![heatmap_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/heat_img.png?raw=true)

Once a heatmap was generated for a particular image, a thresholding function, ```apply_threshold```, was created to set all pixels less than a certain value in the heatmap to zero. The result is an image where only the “hot-spots” are shown. In order to group these assortments of pixels together as distinct cars, the scipy function ```labels``` was implemented. The resulting view can be seen below.

![labels_img]()

To create a visually smoother output image, a final function, ```draw_labeled_boxes``` was implemented to create bounding boxes from the label data. The resulting output image can be seen here.

![new_img](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/readme_images/thresh_img.png?raw=true) 

## Video Processing Pipeline

The heatmap filtering steps were added to the image processing pipeline and I performed a number of tests on image sequences as well as a test video. The results of these tests did fix the issue of generating false positives, but doesn’t generate bounding boxes that completely cover the entire car surface for the duration of the videos. In order to improve the performance of the video processing pipeline, I decided to create a running average of the heatmaps used to filter the false positives. I achieved this by implementing a ```heat``` class as follows:
```
class heat():
    def __init__(self):
        
        self.heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        self.heatmap_list = []
```

By doing this, I was able to smooth the performance of the algorithm and continue tracking the white vehicle through frames that I previously had difficulties finding the car. Additionally, I implemented all three scales (0.75, 1.5, and 2.0) with a threshold of 4. I used the last 5 heatmap images to average the cars founds over multiple frames which prevented the car from completely disappearing. The bounding boxes do become unstable a few times over the entire video, but the cars are tracked through the whole video.

Finally, the following video processing pipeline was applied to the entire project video.

Modified find_cars function:
```
# Define a Function that Finds Cars Quickly
def find_cars_fast(img, svc, X_scaler, scale, ystart, ystop, orient=9, pix_per_cell=8, cell_per_block=2):
    
    scaled_img = img.astype(np.float32)/255
    
    img_tosearch = scaled_img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Create Array for boxes
    box_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg)
            hist_features = color_hist(subimg)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
        
        current_heatmap = add_heat(np.zeros_like(img[:,:,0]).astype(np.float), box_list)
        
    return current_heatmap
```

Modified process_video function:
```
def process_video_fast(img):

    # Generate Heatmaps for Multiple Image Viewing Scales
    heatmap1 = find_cars_fast(img, svc, X_scaler, scale=.75, ystart=400, ystop=528)
    heatmap2 = find_cars_fast(img, svc, X_scaler, scale=1.5, ystart=400, ystop=656)
    heatmap3 = find_cars_fast(img, svc, X_scaler, scale=2.0, ystart=350, ystop=656)

    # Combine Heatmaps
    current_heatmap = heatmap1 + +heatmap2*2 + heatmap3*4
    
    # Add current heatmap to the sum of heatmaps
    heat.heatmap = (heat.heatmap + current_heatmap)
    
    # Clear Left Side of the Image (Filter Other Lane)
    heat.heatmap[:,:600] = 0

    # Append Current Map to List
    heat.heatmap_list.append(current_heatmap)

    # Remove heatmap that is older than 10 frames
    if len(heat.heatmap_list) > 5:
        remove_heat = heat.heatmap_list.pop(0)
        heat.heatmap -= remove_heat

    # Apply thresholding to the heatmap sum
    labelmap = apply_threshold(heat.heatmap, 4)
    
    # Generate labels for the blob images
    labels = label(labelmap)        
    
    # Draw a new Image with bounding boxes
    draw_img = draw_labeled_bboxes(img, labels)

    return draw_img
```

And the resulting video can be found [here](https://github.com/mblomquist/Udacity-SDCND-Vehicle_Detection_and_Tracking-P5/blob/master/project_video_output.mp4?raw=true).

## Discussion

The video processing algorithm performs fairly well throughout the entire video, but the bounding boxes do not always cover the entire area of the white car. Additionally, the bounding boxes generated are quite hectic in nature and seem quite unstable. I believe this can be improved on by integrated a better running average filter to smooth the high-variance bounding boxes from the smaller search windows (scale <= 1.0).

The real-time performance of the video processing algorithm is horrible. The current algorithm takes approximately 30 minutes to complete a 1261-frame, 50-second video. It is far from completing this task in real time. After running a few pipeline bottleneck tests, I noticed that the majority of time taken is during the feature extraction functions of the pipeline. Using a deep learning classification approach may provide significantly improve results as deep learning can be applied to classification with minimal pre-processing of the data during the prediction phase. I believe future work on this project will explore the deep learning route.

## References

 - [1] 	X. Cao, C. Wu, P. Yan and X. Li, "Linear SVM classification using boosting HOG features for vehicle detection in low-altitude airborne videos," 2011 18th IEEE International Conference on Image Processing, Brussels, 2011, pp. 2421-2424. [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6116132&isnumber=6115588)
 - [2] F. Han, Y. Shan, R. Cekander, H. S. Sawhney, and R. Kumar, “A Two-Stage Approach to People and Vehicle Detection with HOG-Based SVM,” [URL](https://pdfs.semanticscholar.org/1c76/6d0f4bf8ff443cbe8a487313e77c20ed4166.pdf)

