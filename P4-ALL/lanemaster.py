import numpy as np
import cv2

class lanemaster():
    
      def __init__(self):

            # Create a Vertical Line that spans the image
            self.ploty = np.linspace(0, 712, 720) 

            self.all = None

            self.leftx = None
            self.lefty = None
            
            self.rightx = None
            self.righty = None

            self.left_fit = None
            self.right_fit = None

            self.weights_left = None
            self.weights_right = None

            self.window_margin = 100

            pass

      def window_search(self,img_warped):

            # Take a histogram of the bottom half of the image
            histogram = np.sum(img_warped[img_warped.shape[0]/2:,:], axis=0)

            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((img_warped, img_warped, img_warped))*255
            
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            
            # Set height of windows
            window_height = np.int(img_warped.shape[0]/nwindows)
            
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = np.array(img_warped.nonzero())
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            
            # Set the width of the windows +/- margin
            margin = 100
            
            # Set minimum number of pixels found to recenter window
            minpix = 50
            
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):

                  # Identify window boundaries in x and y (and right and left)
                  win_y_low = img_warped.shape[0] - (window+1)*window_height
                  win_y_high = img_warped.shape[0] - window*window_height
                  win_xleft_low = leftx_current - margin
                  win_xleft_high = leftx_current + margin
                  win_xright_low = rightx_current - margin
                  win_xright_high = rightx_current + margin
        
                  # Draw the windows on the visualization image
                  cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                  cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        
                  # Identify the nonzero pixels in x and y within the window
                  good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                  good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
                  # Append these indices to the lists
                  left_lane_inds.append(good_left_inds)
                  right_lane_inds.append(good_right_inds)
        
                  # If you found > minpix pixels, recenter next window on their mean position
                  if len(good_left_inds) > minpix:
                        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                  if len(good_right_inds) > minpix:        
                        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            # Generate x and y values for plotting
            ploty = np.linspace(0, img_warped.shape[0]-1, img_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            self.leftx = leftx
            self.lefty = lefty

            self.rightx = rightx
            self.righty = righty

            self.left_fit = left_fit
            self.right_fit = right_fit
    
            return ploty, left_fitx, right_fitx, out_img

      def filter_search(self,b_warp):

            # Find Pixel Locations in Binary Image
            nonzero =  b_warp.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            margin = self.window_margin
    
            left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) 
                              & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 
            right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) 
                               & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
    
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            self.leftx = np.hstack((self.leftx,leftx))
            self.lefty = np.hstack((self.lefty,lefty))

            self.rightx = np.hstack((self.rightx,rightx))
            self.righty = np.hstack((self.righty,righty))

            # Filter the total pixels based on the latest fit
            leftx, lefty, rightx, righty = self.filter_pixels(self.leftx,self.lefty,self.rightx,self.righty)

            # Polyfit Data
            self.left_fit = np.polyfit(lefty,leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

            left_size = int(len(self.leftx)/4)
            right_size = int(len(self.leftx)/4)

            if len(self.leftx) > 100000:
                  self.leftx = np.delete(self.leftx,range(left_size))
                  self.lefty = np.delete(self.lefty,range(left_size))
            if len(self.rightx) > 100000:
                  self.rightx = np.delete(self.rightx,range(right_size))
                  self.righty = np.delete(self.righty,range(right_size))

            left_fitx = self.left_fit[0] * (self.ploty ** 2) + self.left_fit[1] * self.ploty + self.left_fit[2]
            right_fitx = self.right_fit[0] * (self.ploty ** 2) + self.right_fit[1] * self.ploty + self.right_fit[2]

            return self.ploty, left_fitx, right_fitx 


      def filter_pixels(self,leftx,lefty,rightx,righty):

            # Set Left Boundary Based on Fit Data - Full Range
            left_indx_full = ((leftx > (self.left_fit[0] * ( lefty ** 2 ) + self.left_fit[0] * lefty + self.left_fit[2] - self.window_margin )) 
                              & (leftx > (self.left_fit[0] * ( lefty ** 2 ) + self.left_fit[0] * lefty + self.left_fit[2] - self.window_margin )))

            # Set right Boundary Based on Fit Data - Full Range
            right_indx_full = ((rightx > (self.right_fit[0] * ( righty ** 2 ) + self.right_fit[0] * righty + self.right_fit[2] - self.window_margin )) 
                              & (rightx > (self.right_fit[0] * ( righty ** 2 ) + self.right_fit[0] * righty + self.right_fit[2] - self.window_margin )))

            leftx = leftx[left_indx_full]
            lefty = lefty[left_indx_full]

            rightx = rightx[right_indx_full]
            righty = righty[right_indx_full]

            return leftx, lefty, rightx, righty



