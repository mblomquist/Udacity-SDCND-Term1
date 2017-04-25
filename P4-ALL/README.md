# Udacity-SDCND-Advanced_Lane_Detection-P4
Advanced Lane Detection - Project 4

## Project Description

This project builds an advanced lane-finding algorithm using distortion correction, image rectification, color transforms, and gradient thresholding to create a lane area map. Additionally, the project outputs the lane curvature and vehicle displacement.

## TODO List:
- [ ] Finish python implementation
- [ ] Smooth the lines
- [x] Add project description.

## Project Notebook

This project was created using Jupyter Notebook and the link to the file can be found here:

[Project 4 Notebook](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/Project4.ipynb)

The sections referenced in the writeup below correspond to the sections of the Project 4 notebook. To reference the explicit code for a particluar step, please look at the appropriate section in the Project 4 notebook.

## Camera Calibration (Section 2)

### Computing Chessboard Corners (Section 2.1.1)

In order to compute the camera matrix, I started by loading in the various calibration images of a chessboard. The images were provided by Udacity for this project. Once I had loaded the calibration images, I created a set of empty arrays for the object and images points. The object points represent the corners of an evenly spaced grid and the image points represent the corners found in the calibration images. Next, I used a for loop to cycle through each of the calibration images and append the list of corner points to the image point array. In order to verify I was finding the appropriate points in the images, I printed out a test example which is shown here:

![chessboard](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/chessboard.png?raw=true)

### Computing the Camera Matrix and Distortion Coefficients (Section 2.1.2)

Once I had found the chessboard corners, I used the OpenCV function, ```calibrateCamera``` , to calculate the camera matrix and the distortion matrix. I next assigned the camera matrix and distortion matrix to global vairables so that they could be used in each of the functions that I call later in the image and video processing pipelines. In order to verify that I computed the coefficients correctly, I printed an example of an input calibration image and its undistorted version. That example is shown here:

![chessboardundistort](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/chessboard_undistort.png?raw=true)

## Image Processing Pipeline (Still Images)

### Distortion Correction (Section 3.1)

The image processing pipeline starts be undistorting the test images which is taken with the same camera as the calibration images. I use the same set of functions as mentioned in the camera calibration section. To verify the images were processed correctly, I printed an example of the distorted and undistorted images. The example is shown below and the difference in distortion is most noticeable at the bottom left corner and bottom right corner of the image.

![testundistort](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/test_undistort.png?raw=true)

### Creating a Thresholded Binary Image (Section 3.2 & Section 3.3)

In order to create a thresholded binary image, I utilized color and gradient thresholding. For the color threshold, I started by converting the images to the HSV color specturm (shown below).

![hlsconversion](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/hsv_grad.png?raw=true)

Next, I extracted the S-channel of the HLS image as this channel ended up creating the best result.

![schannelextraction](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/schannel_extraction.png?raw=true)

With the extracted S-channel image, I performed a color threshold to create a binary image. The progression from the undistorted image to the S-Channel binary image is shown below.

![schannelbinary](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/sbinary.png?raw=true)

For the gradient threshold image, I converted the undistorted input image to a HLS image, extracted the L-channel, and used the sobel operated to take the gradient in the x direction. Since lane lines are mostly vertical, I wanted to find the color difference along the x-axis of the image. I then used thresholding on the gradient image in order to create a binary version. 

![lgrad](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/lgrad_binary.png?raw=true)
I then combined the gradient binary with the s-channel binary to create a combined binary image.

![combinedbinary](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/binary_creation.png?raw=true)

### Perspective Transform (Section 3.5)

In order to perform a perspective transform or bird's eye view, I first chose the particular points of interest manually by drawing lines in the undistorted image. 

![points](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/transform_lines.png?raw=true)

Next, I use the selected points in an array along with a set of points that I want to map the image to. I used the following matrices where src represents the source points and new represents the mapped image points.

```
new_mtx = np.float32([[260, 0],
                      [1040, 0],
                      [1040, 720], 
                      [260, 720]])

src_mtx = np.float32([[580,460],
                      [700,460],
                      [1040,680], 
                      [260,680]])
```

The resulting transformed images are shown here:

![birdseye](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/birdseye.png?raw=true)
![birdseye2](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/cb1.png?raw=true)


### Finding Lane Line Pixels (Section 3.5)

Now, with the bird's eye image, I check the location of the non-zero pixels in the image to find the lane lines. As a quick sanity check, I use a histogram to find where the non-zero pixels are in the bottom half of the image.

![histogram](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/histogram.png?raw=true)

Next, I implement a sliding window scheme to collect all of the non-zero pixel values in the image. I chose to implement 9 windows for each line (left and right) as this value tended to produce good results. Once all of the pixels were collected, I used a 2nd order polynominal fit to create a line the matched the left and right lane lines. An example output is shown here:

![slide](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/sliding_window.png?raw=true)

To make the code run a bit quicker next time around, I created a slightly different verion of this method. I used the previously found lane lines to set up a window to look for the new lane lines. This became useful when doing video processing. I've included an example of that output here:

![window](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/quicksearch_window.png?raw=true)

### Calculating the Curvature of a Lane Line and Position of the Vehicle (Section 3.6)

Once I had a polynominal fit of the left and right lane lines, I measured the radius of curvature of the lines using the method described [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). The measurement was corrected to real world values using a standard conversion from pixel to meters (3.7/700 in the x-direction and 30/720 in the y-direction). For one of the test images, the following values were calculated:
```
Left Curvature:  2790.58232173 m
Right Curvature:  103018.133088 m
Averaged Curvature:  52904.357705 m
```
I also measured the distance of the camera from the center of the lane by taking the intercept points of the left and right lane line fits and comparing them with the center of the image. The measurement from the same test image as above came out to the following:
```
Lane Center:  -0.136374540656 m
```

### Image Pipeline Output (Section 3.7 & Section 3.8)

Finally, I prepared the output image by creating a polyfit of the left and right lane line calculations and imposing it onto the undistorted image. I also added text displaying the measured road curvature and the distance from center.

![outimg](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/output_image.png?raw=true)

To make everything run smoothly when processing multiple images, I created a set of functions for each of the operations described above and called them in the image processing pipeline as follows:
```
def process_image(img):
    
    # Distortion Correction
    img = cv2.undistort(img, c_mtx, d_mtx, None, c_mtx)
    
    img_binary = combine_thresh(img)
    
    img_warped, Minv = warp(img_binary)
        
    ploty, left_fitx, right_fitx, left_fit, right_fit, img_out = find_lanes(img_warped)
    
    left_curverad, right_curverad = calc_curve(ploty, left_fitx, right_fitx)
    
    img_out = create_output(img, ploty, left_fitx, right_fitx, left_curverad, right_curverad, Minv)
    
    return img_out
```

To test the pipeline, I ran 7 images through it. Those are shown here:

![op1](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/op1.png?raw=true)
![op2](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/op2.png?raw=true)
![op3](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/op3.png?raw=true)
![op4](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/op4.png?raw=true)
![op5](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/op5.png?raw=true)
![op6](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/readme_examples/op6.png?raw=true)

## Video Processing (Section 4)

The pipeline for processing the video image was nearly identical to that previously described. However, the exceptions were based on using the quicker, line search method. Furthermore, I implemented a check to validate the polyfit outputs. I've provided an example of the pipeline here:

```
def process_video(img):
    
    # Distortion Correction
    img = cv2.undistort(img, c_mtx, d_mtx, None, c_mtx)
    
    img_binary = combine_thresh(img)
    
    img_warped, Minv = warp(img_binary)
        
    if line.detected == False:
        ploty, left_fitx, right_fitx, left_fit, right_fit, out_img = find_lanes(img_warped)
        
        line.fit_left = left_fit
        line.fit_right = right_fit
        
        line.fitx_left = left_fitx
        line.fitx_right = right_fitx
        
        line.detected = True
        
    else:
        ploty, left_fitx, right_fitx, left_fit, right_fit = update_lanes(img_warped, line.fit_left, line.fit_right)
        
        if abs(left_fit[0]) > 5*abs(line.fit_left[0]):
            
            weights_left1 = np.zeros_like(line.fitx_left)+1.5
            weights_left2 = np.zeros_like(left_fitx)+.5
            
            weights_right1 = np.zeros_like(line.fitx_right)+1.5
            weights_right2 = np.zeros_like(right_fitx)+.5
            
            left_fitx = np.hstack((line.fitx_left,left_fitx))
            weights_left = np.hstack((weights_left1,weights_left2))
            
            right_fitx = np.hstack((line.fitx_right,right_fitx))
            weights_right = np.hstack((weights_right1,weights_right2))
            
            ploty = np.hstack((ploty,ploty))
            
            left_fit = np.polyfit(ploty,left_fitx,2,w=weights_left)
            right_fit = np.polyfit(ploty,right_fitx,2,w=weights_right)
            
            line.fit_left = (left_fit + 24*line.fit_left)/25
            line.fit_right = (right_fit + 24*line.fit_right)/25
            
            left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]
            
        else:
            
            line.fit_left = (left_fit + 24*line.fit_left)/25
            line.fit_right = (right_fit + 24*line.fit_right)/25
            
            line.fitx_left = left_fitx
            line.fitx_right = right_fitx
            
    left_curverad, right_curverad = calc_curve(ploty, left_fitx, right_fitx)
    
    img_out = create_output(img, ploty, left_fitx, right_fitx, left_curverad, right_curverad, Minv)
    
    return img_out
```

### Project Video (Section 4.4)

The link the the video can be found here:

[Project Video](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/project_video_output.mp4?raw=true)

### Challenge Videos (Section 4.5 & 4.6)

The image processing pipeline did not perform as well on the challenge videos. In the first challenge video, there were a few places that resulted in a loss of the lane line. In the harder challenge video, there were numerous areas that the detected lane was off. This will be an area for improvement on future iterations of the code.

[Challenge Video](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/challenge_video_output.mp4?raw=true)

[Harder Challenge Video](https://github.com/mblomquist/Udacity-SDCND-Advanced_Lane_Detection-P4/blob/master/hard_challenge_video_output.mp4?raw=true)

## Project Discussion

Throughout the project, I used fairly standard methods to extract the lane features. Using only the HSV, color and gradient threshold methods resulted in fairly accurate results. However, these methods didn't hold up as well on the challenge videos. Exploring with other thresholding methods may prove to be more useful in future iterations of the project and perform better then there are drastic differences in the road conditions (as in the challenge videos). This would be especially helpful for roads similar to that of the challenge video, where there is different colored road partitions.

The chosen perspective matrix points worked very well with fairly straight lanes. In the harder challenge video, the prespective transform was likely the cause of the breakdown in the image processing pipeline. With sharp turns (small radius of curvature), the perspective transform would loose a lot of lane information as it looks in the wrong places. Using a variable perspective transform may be the ideal way to find the lanes in those types of conditions.
