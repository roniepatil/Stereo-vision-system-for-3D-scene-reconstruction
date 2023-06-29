# Stereo vision system for 3D scene reconstruction

## Author
Rohit M Patil

## Description
This project focuses on implementation of concept of Stereo Vision. With given 3 different datasets, each of them contains 2 images of the same scenario but taken from two different camera angles. By comparing the information about a scene from 2 vantage points, we can obtain the 3D information by examining the relative positions of objects.

## Results
* **SIFT**
  -  ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/SIFT_on_pendulum.png)
  -  ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/SIFT_on_octagon.png)
  -  ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/SIFT_on_curule.png)
* **Epipolar lines**
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/epipolar_lines_on_rectified_images_pendulum_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/epipolar_lines_on_rectified_images_octagon_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/epipolar_lines_on_rectified_images.png)
* **Disparity heatmap**
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Disparity_in_heatmap_on_pendulum_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Disparity_in_heatmap_on_octagon_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Disparity_in_heatmap_on_octagon_dataset.png)
* **Disparity greyscale**
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Disparity_in_greyscale_on_pendulum_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Disparity_in_greyscale_on_pendulum_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Disparity_in_greyscale_on_curule_dataset.png)
* **Depth heatmap**
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Depth_in_heatmap_on_pendulum_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Depth_in_heatmap_on_octagon_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Depth_in_heatmap_on_curule_dataset.png)
* **Depth greyscale**
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Depth_in_greyscale_on_pendulum_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Depth_in_greyscale_on_octagon_dataset.png)
  - ![alt text](https://github.com/roniepatil/Stereo-vision-system-for-3D-scene-reconstruction/blob/main/Images/Depth_in_greyscale_on_curule_dataset.png)


## Dependencies

| Plugin | 
| ------ |
| tqdm | 
| numpy | 
| cv2 | 
| matplotlib | 
| time | 

## Instructions to run


**1) Curule Dataset:**
```bash
python curule_solution.py
```
or
```bash
python3 curule_solution.py
```

**2) Octagon Dataset:**
```bash
python octagon_solution.py
```
or
```bash
python3 octagon_solution.py
```


**3) Pendulum Dataset:**
```bash
python pendulum_solution.py
```
or
```bash
python3 pendulum_solution.py
```