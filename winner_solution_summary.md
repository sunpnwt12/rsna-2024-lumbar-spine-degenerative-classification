# Winners' Solutions
## 1st place
- 2 stage approach
    - 1st stage is instance_number prediction and coordinate prediction
    - 2ns stage is severity prediction
    - meaning that there are 3 types of models

- 1st stage
    - instance_number prediction
        - Used 3D ConvNeXt to prediction instance_number for each level
        - image sized to 512x512x32
        - dicom sorted by metata padded to 32 depth
    - coordinate predition
        - simple regession model predicting relative coordinate

    - Axial instance_number and coordinate
        - use hengch32 public kernel
        - coordinate prediction is similar to sagittal

- 2nd stage
    - used 2.5d model and MIL (MIL was better at last)
    - Preprocessing
        1. pick up 5 images centered by target slices
        2. reshaped 512x512
        3. cropped using coordinates 96px to left, 32px to right, 40px to upper and lower
            - this is only examples of the certain condition
            - each condition has its own certain area that was determined by the solution's owner
    - Data augmentation
        - before crop
            - random shift coordinates by +/-10
            - random shift instance_number by +/-2 (VERY IMPORTANT FOR ROBUSTNESS)
        - after crop (very minimal augment)
            - random brightness contrast
            - shiftscalerotate
    - Modeling
        - input is 5 intances of 1 channel (bs, 5, 1, h, w)
        - feed to backbone, output as (bs, 5, features)
        - then put through bi-lstm, output as (bs, 5, features)
        - main head
            - attention-based weighted mean (bs, features)
                - for SCS, using used 2 backbone concatnated then forward to FC head
            - predicting severity 3 classes (bs, 3)
        - aux head
            - predicting which slices is the target slices (bs, 5)

## 2nd place *(NOT DONE)*
### YUJIARIYASU's Part
- Main approach is ensembling of small models.
    - worked separately on different planes
    - each model predict each condition targeting only on severities

- Worked seperately on Sagittal and Axial
    - Sagittal
        1. classify suitable slices
        2. estimate regions (levels) within the images
        3. Classification input is 5 images (MIL model accept 5 images)>
            - backbone is ConvNext small
            - for spinal and subarticular
                - used 5 images from the center of the series
            - for foraminal
                - used 5 images centered in-between of spinal and subarticular
            - There were some use T1 and T2 put into separate channels
                - if the study does not have one or another using only 1
    - Axial
        1. calculate which slice is belong to which level
        2. train YOLOX finding the regions (coordinates?) of severities
        3. Classification (maybe 1 images?)
            - backbone is ConvNext small
            - For spinal related predictions
                - use the direct regions
            - For non-spinal
                - use left and right of the image to separate sides
                
- Noise reduction
    - removed sample with high loss
    - excluded samples where there is huge different between GT and predictions more than 0.8

- *This is only brief solution, comeback to finish later*

### Bartley's part
- Summary
    - 2-stage approach using sagittal only
    - cropped location using 2 point of coordinates at both end of disc.
    - Spinal and subarticular labels are inferenced from T2 and foraminal from T1
    - using many techniques like TTA, Pseudo labelling, Image augmentation

- 1st stage
    1. predict location (coordinate) of each disc
    2. crop the area of each disc using coordinate
    3. for crop size
        - the height is determined with average distance between levels
        - the width is a bit different for each planes
            - T1 is using the same distance as the height
            - T2 is taken from average disc width
                -  (this solution using 2 points of coordinates. at two end of disc)

- 2nd stage
    - classify labels for each cropped sequence which equally treated from each levels
    - denoising method was calculating half from given labels and pseudolabels during training
    - for T1
        - used middle 24 frames to encoder pass to LSTM then pooled with Attention
    - for T2
        - process is similar but using only 5-16 frams and different sizes for increasing the diversity purpose

- During inference
    -  nine different rotations TTA

- Worked on Axial but did not improved overall team CV

### Ian's part

- Solution break into 2 stages

- Stage 1 part 1 Finding Slice
    - trained using CNN-Transfromer model by taking 3D sagittal T1 series

    - Sagittal T1 weighted (Foraminal)
        - predicting 20 targets
            - 10 targets for distance ImagePatientPatient0 from target foramen each level each sides
            - 10 targets for classification label for target foramen slice 1 for the target and 0.5 as adjacent slices each level
            - optimizer was combination of smooth L1 and BCELoss.

    - Sagittal T2 weighted (Spinal)
        - same as T1, but 5 targets for each level, and other 5 for target slice

    - Axial T2 (Subarticular)
        - Assigned to 11 labels either intervertebral level (l1/l2, l2/3, ..) or in-between intervertebral level (l1, l2, ...) using grouth truth coordinates
        - if it was above l1 or below s1, then assign it to its most end
        - left and right subarticular were added as 2 more classes
        - along with 10 targets for the distance (ImagePositionPatient2)
        - used public code mapping Axial location from Sagittal series

    - Axial T2 (Spinal)
        - Took average of left and righ subarticular slices
            - the results were the same or off by 1 slice

- Stage 1 part 2 Finding Keypoint
    - trained regression model prediction keypoint (coordinates)
    - Axial spinal keypoint location is average value of left and right multiplied by 1.05 as the ground truth usually a bit lower than subarticular zone.


- Stage 2 Classification
    - Generated cropped images using the ground truth cooridnates.
    - Input image number is 1 image per channel, target image is at the center ,and others are its adjacent slices.

- Handling edge cases
    - Multiple Axial T2
        - use crops from both
    - Missing Axial T2
        - use predition from only spinal sagittal model
            - (typically, in most case, the prediction of spinal canal stenosis was from sagittal and axial)
    
## 3rd place

- Summary
    - stage 1
        - trained keypoint detection
        - sagittal crop each disc level
        - assign axial slice level using sagittal then crop

    - stage 2
        - use center classifier for spicnal canal stenosis
        - use side classifier for neural foraminal narrowing and subarticular stenosis
            - left and right is split in prior
            - so model only have to predict severity
        - each series has its own backbone after global average pooling concat them altogether
            - then pass through transformer encoder forcing model to learn slices relationship
            - at the end, global average pooling then fc

- Validation Strategy
    - y: Number of moderate or higher severity cases included in one study'
    - groups: study_id

- Cross-entropy weight [1.0, 2.0, 4.0]
- Temperature Scaling

## 8th place *(NOT DONE)*

- the main idea of solution is 2-step of the model
    - 2D classifer
        - take 2d images (full images, without any crops)
        - predict 5 conditions x 5 levels x 3 class
        - use weight average instead of using global average pooling
            - theese weights are coming from sub task that predicting ROI of each level
            - Advantages
                - model can consider overall context of the images
                - less sensitive to ambiguities of cropping and detection

    - 1D classfier
        - stack outputs frompre vious 2d classifier

- Relabeling
    - as there were many inconsistencies in the labels (the keypoints)
    - noise reduction

## 11th place solution
### Overview
- 2 stages approach
    - 1st stage
        - estimate slice index associated with the condition
        - crop ROI
    - 2nd stage
        - Each condition is classified by an independent model
        - 3 main condition (left-right are united as one condition) concatenated into the final submission
        - the final model number are around 4-7 models per condition
        - the weighted averaged of each model is optimized by nelder-mead.

### Yumeneko's part
- Slice index estimation
    - Sagittal T2
        - used exactly the middle slice of series
    - Sagittal T1
        - for left: number_of_slices * 0.274
        - for right: number_of_slices * 0.719
    - Axial T2
        - used public kernel by Ian

- Keypoint prediction
    - simple regression model predcitiong relative coordinates

- Image cropping
    - 3 types of viewpoint were used in different models
        1. Sagittal overall crop
            - crop all level from l1_l2 to L5_s1
        2. crop corresponding ROI of the level
            - the are was determined by distance betwen y of target of the its above and below y
        3. Axial crop
            - crop at the averaged x-point estimated from each sides

- Classification
    - Used 2.5d input where range is between 3 to 5 slices around the determined target slice
    - each model take series that corresponding to labelled condition
        - Spinal canal stenosis used Sagittal T2
        - Neural foraminal narrowing used Sagittal T1
        - Subarticular Stenosis used Axial T2
    - image sized to 512
    - randomly increase or decrease target slice index by +/-2 as augmentation
    - training used GT keypoint. However, in CV calculation, predicted  keypoint were used

    - Details on each type of models
        - Spinal canal stenosis
            - using only Sagittal T2
            - used LSTM at the head
        - Neural foraminal narrowing 
            - right and left were not distinguished
        - Subarticular Stenosis
            - used only level-cropped images