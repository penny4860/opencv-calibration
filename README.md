## Camera Calibration with OpenCV

#### 1. Get image points

Get image points from several checkerboard images

* inputs
	* image
	* n_pattern_rows
	* n_pattern_cols

* outputs
	* image points : 2d image coordinate points of checkerboard corners


#### 2. Get object points

Get object points from several checkerboard images

* inputs
	* n_images
	* n_pattern_rows
	* n_pattern_cols
	
* outputs
	* object points : 3d world coordinate points of checkerboard corners


#### 3. Do calibration for test image size

```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
```

#### 4. Distortion correction using camera matrix & distortion coefficient

```
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

