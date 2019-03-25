# PointRCNN
Warning: This is **not** the official implementation of PointRCNN, and it is still in progress.
## Introduction
A 3D object detector that takes point cloud and RGB image(optional) as input.

## Architecture
1. Perform foreground point segmentation on the whole point cloud
2. Output a 3D proposal box for every foreground point
3. Crop point cloud with proposal boxes and feed into the 2nd-stage classification and box refinement network
![](images/architecture2.png)

## Evaluation
### Recall of RPN
|    Method  | Avg. Recall(IOU>0.5)|
| ---------- | ------------------- |
| Point Only |                 81% |
| Point+Image|                 86% |

### Final detection mAP

|    Class   | 3D mAP(Easy, Moderate, Hard)  | BEV mAP(Easy, Moderate, Hard)  |
| ---------- | ----------------------------- |--------------------------------|
| Car        | 62.179321, 57.947697, 60.453468 |81.649628, 75.761436, 76.957726|
| Pedestrain | 59.891392, 61.954231, 54.722935 |73.589073, 67.023071, 67.218903|
| Cyclist    | 69.380432, 51.198471, 43.347675 |71.138779, 52.781166, 44.486042|



## Results
![](images/001101.png)
![](images/001138.png)

## Todo List
- Use segmentation result from RPN to help ROI pooling
- Use dense points obtained from depth completion/stereo for 2nd-stage network
