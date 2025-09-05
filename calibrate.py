'''
SOURCE: https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
'''

# Import required modules
import cv2
import numpy as np
import glob
 
CHESSBOARD = (6, 9)
images = glob.glob('chessboard/*.JPG')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
points = np.zeros((1, CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
points[0, :, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
coordinates_3d = []
coordinates_2d = []
 
for img in images:
    image = cv2.imread(img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    ret_val, corners = cv2.findChessboardCorners(gray_image, CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
 
    if ret_val == True:
        coordinates_3d.append(points)
 
        corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
        coordinates_2d.append(corners2)
 
        image = cv2.drawChessboardCorners(image, CHESSBOARD, corners2, ret_val)
 
        cv2.imshow('img', image)
        cv2.waitKey(1000)
 
cv2.destroyAllWindows()
 
retVal, final_matrix, displacement = cv2.calibrateCamera(coordinates_3d, coordinates_2d, gray_image.shape[::-1], None, None)

with open('camera_values.txt', 'w') as f:
    f.write(str(final_matrix[0][0]))
    f.write("\n")
    f.write(str(displacement[0][0]))
    f.write("\n")
    f.write(str(displacement[0][1]))


