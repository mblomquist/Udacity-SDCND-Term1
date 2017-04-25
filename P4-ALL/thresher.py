import numpy as np
import cv2

class thresher():

      def __init__(self):

            pass

      def combined_thresh(self,img):
          # Define Thresholding Max and Min Condition
          gradient_binary = self.gradient_thresh(img)
          hsv_binary = self.hsv_thresh(img)
          hls_binary = self.hls_thresh(img)

          combined_binary = np.zeros_like(img[:,:,0])
          combined_binary[( (gradient_binary == 1) | (hsv_binary == 1) | (hls_binary == 1) )] = 1

          return combined_binary

      def gradient_thresh(self,img):

          # Convert Image to HLS Color Space
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
          # Strip L-Channel Information
          img = img[:,:,1]

          # Use Sobel Function to Take the X-Direction Gradient
          sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=-1)
          sobelx = np.absolute(sobelx)

          sobel = (255*sobelx/np.max(sobelx))

          # Threshold the Gradient
          thresh_min = 30
          thresh_max = 100

          gradient_binary = np.zeros_like(sobel)
          gradient_binary[(sobel >= thresh_min) & (sobel <= thresh_max)] = 1

          return gradient_binary

      def hsv_thresh(self,img):

          # Convert Image to HSV Color Space
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

          ''' Note: This idea came from Mehdi Sqalli here:
                  https://medium.com/@MSqalli/advanced-lane-detection-6a769de0d581#.ta6un32x9 '''

          # Define Thresholding Max and Min Conditions (Yellow)
          yellow_min = np.array([15,100,120], np.uint8)
          yellow_max = np.array([80,255,255], np.uint8)

          # Define Thresholding Max and Min Conditions (White)
          white_min = np.array([0,0,200], np.uint8)
          white_max = np.array([255,30,255], np.uint8)

          # Create Masks for Yellow and White Lines
          yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
          white_mask = cv2.inRange(img, white_min, white_max)

          # Create Binary Output
          hsv_binary = np.zeros_like(img[:,:,0])
          hsv_binary[((yellow_mask != 0) | (white_mask != 0))] = 1

          return hsv_binary

      def hls_thresh(self,img):

          # Convert Image to Lab Color Space
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

          # Define Thresholding Max and Min Conditions (White)
          thresh_min = 250
          thresh_max = 255

          # Strip b-channel values
          img = img[:,:,2]

          # Create Binary Output
          hls_binary = np.zeros_like(img)
          hls_binary[(img > thresh_min) & (img <= thresh_max)] = 1

          return hls_binary

