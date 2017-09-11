import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def get_object_points(n_images, n_pattern_rows=6, n_pattern_cols=8):
    point = np.zeros((n_pattern_rows*n_pattern_cols,3), np.float32)
    point[:,:2] = np.mgrid[0:n_pattern_cols, 0:n_pattern_rows].T.reshape(-1,2)

    points = [point for _ in range(n_images)]
    return points


def get_image_points(images, n_pattern_rows=6, n_pattern_cols=8, show_pts=False):
    imgpoints = [] # 2d points in image plane.
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (n_pattern_cols,n_pattern_rows), None)
    
        # If found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            
            if show_pts:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (n_pattern_cols,n_pattern_rows), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
     
    cv2.destroyAllWindows()
    return imgpoints


# Make a list of calibration images
images = glob.glob('calibration_wide/GO*.jpg')
img_points = get_image_points(images, n_pattern_rows=6, n_pattern_cols=8)
obj_points = get_object_points(len(img_points), n_pattern_rows=6, n_pattern_cols=8)

# Test undistortion on an image
img = cv2.imread('calibration_wide/test_image.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()

