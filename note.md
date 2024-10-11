# 7.7

- handle null by fill with normal_mild condition
    - do not know how it would go, just use this for now

# 7.8

- 25 classes
- predict 0 to 1 probabbility ot the patient
    - across 3 grades (3 classes per condition_level)
        - normal/mild
        - moderate
        - servere

- three conditions need to asses
    - 5 levels for each condition
        - each vertebral disc between levels

    - Neural Foraminal narrowing (Left & Right)
        - 10 classes

    - Subarticular stenosis (Left & Right)
        - 10 classes

    - Spinal Canal Stenosis
        - 5 classes

- MRI
    - 3 planes
        - coronal plane (not in dataset)
        - axial plane
            - (horizontal) top -> bottom
            - perpendicular to the spine

        - sagittal plane
            - (vecical) left -> right
            - parrel to the spine
    - variants
        - T1 weighted
            - show fat bighter
            - inner part of bones brighter
        - T2 weighted
            - water brigher
            - spinal canal brighter

- Correspondence
    - Neural Foraminal narrowing (Left & Right)
        - using Sagittal T1

    - Subarticular Stenosis (Left & Right)
        - Axial T2
    
    - Spinal Canal Stenosis
        - using Sagittal T1/T2


- solution from the similar past competiion
    - 1st place
        - 3D semantic segmentation to create mask
        - train 2.5d
            - train 2dconv as encoder and LSTM at the head


- first let's try create segmentation pipeline
    - the given dataset does not contain groundtruth info
    - found dataset that cn use to train segmentation for lumar spine
        - train png split by group of patient based on either segmentation exists or not?
        - pipeline is not finished


# 7.9

- Is this segmentation model multilabel?

- Is it better the use 3D segmentation instead of 2D?

- What to do next?
    - train segmentation on zenodo
        - not sure 3 channels or 1 channel is better
        - try 3 channels first then 1 channel later

- found bugs when loss become negative
    - don't know how to fix


# 7.10

- bugs: loss become negative, what can cause this?
    - loss function
    - inputs and targets
    - fixed: mask and output need to dvided by 255.0
        - also found that ToTensorV2 can tranpose mask

- when inferencing, have to load image directly from dicom.

# 7.14

- solution from the past competition (RSNA2022)
    - 8th place solution
        - classify each vertebrae using segmentation
        - crop ROI
        - classify in slice-level
        - RNN extraced features at the end of the classifier


- SPDIER Dataset seems fine to use
    - it contains btoh Lumber spine masks and intervertebral discs


- Cause of each symptoms
    - Canal Stenosis
        - bulging disk
        - bony osteophytes (outgrowths of the vertebral bodies due to degenerative changes)
        - ligamental thickening (of the ligaments that run along the length of the spinal canal)
    
    - Subarticular Stenosis
        - compression of the spinal scord in the subarticular zone
        - can cause by herniated disk or degneerative disk

    -

- 2.5D segmentation
    - 3 channels
        - -1 slice
        -  0 slice 
        - +1 sclie

- create mask using coordinate
    - draw square over coordinate
    - use it as axu head

- maybe segmentation is better because the past competition top solution used them even they were given coordinate

- tasks for tmr
    - create 3 stripe at in create_spider notebook


# 7.15

- should I use only bone?
    - disks seem important too.
    - what about spinal canal

- So, intially, go with lumbar, and in-between disc first (excluding spinal canal)
    - separate two models
        - one for lumber spine
        - one for disc


- In SMP document, DiceLoss needs each class to be in different channels of its own
    - the segmented section is 1 and others is 0
    - in this case, 6 channels is needed
        - 0 for background
        - 1 to 5 are for L5 to L1, repectively

- task for tmr
    - fix bug in create_spier_spine
        - L1 (label 5) is not found anywhere 
        - found bug
            - accidentally, covert to np.uint8 without scale to 255. (it was 0-1 range before convert)
    - maybe try combined all thoracic spine into 1 label instead of filter out to backgroud 
        - done

# 7.16

- Got a decent performance from segmentation model
    - inferencing is ready
    - need to re-thinking about its necessity

- do I really need multilabel mask?

- task for tmr
    - explore a bit more about binary mask and multilabel mask as a aux loss head

# 7.17

- stenosis features can be observed via
    - narrowing of spaced
        - easily observed in axial plane
    - disc changes
        - disc bulding or herniation
        - loss of disc height
    - bone changes
        - outgrowths bone

- CT Scan vs MRI
    - CT Scan suits for imaging Bones, Organs
    - MRI suits for imaging Muscle, ligaments, tissue

- In this competition is using MRI
    - meaning that segmenting bones is harder
    - putting effort on disc segmenting might be better?

- swtiched model from efficient-b0 to tf_efficientnet_b0.ns and combo loss over dice
    - score become so much better 

- feel like there is a bug, because both spine and disc segmentation do not have 1st class 

- task for tmr
    - explore usage of the segmentation


# 7.18

- L5 is still missing in segment
    - label 1 became L4
    - turned out label 5 (indexing-0) became thoracic spine
    - found and fixed bug
        - forget to +1 to c in postprocess
            - it made class 1 (L5) disappear because it was in the first channel

- Instead of using SPIDER dataset, I can use the given coordinates
    - by create a windows expanding from the the coordinates
    - cheack first if all image has coor


- after get complete segmention of spine
    - crop them with pair of 2 classes to represent area in-between
    - ex. crop L1 and L2 height is from L1 top and L2 bottom,
    width is from both pair of classes aligning with a bit of space

- task for tmr
    - explore way to crop them

# 7.19=

- How should I pick samples?
    - pick samples evenly based on quantitle start from 0.05 to 0.95 of slices?
    - 

- cropping process (sagittal)
    - crop between 2 classes which will expose disc in-between
    - crop from 2.5D MRI image which is 3 channels
    - that make up to 15 channels containing 3 channels of each level (ex.l1/l2, l2/3, etc.)
        - L1/L2 -> 3 chans
        - L2/L3 -> 3 chans
        - L3/L4 -> 3 chans
        - L4/L5 -> 3 chans
        - L5/S1 -> 3 chans
    - so the final image shape would be (bs, 15, h, w)
    - when cropping, It might be better to considered z-axis (channels dim) as the most important element
        - crop voxel: 
            - find all min and max of x, y of each in-between level
            - crop based on that min and max of that in-between level regard less of the individual min or max for consistency across the series when creating 2.5D image
    
- problem found
    - segmentation using lumbar spine contained only L1-L5 that means it does not have S1
    - but disc segmentation does
    - can solve by manually creating the bound boxes

- I still need axial plane
    - perhaps axial is more effective than sagittal
    - 2022 competition was using Cervial not Lumbar
        - might not work, haven't tried yet

- for better quality I should predict segmentation mask on dcm directly

- In training step, there are a few slice that was given label and cooordinates
    - but in inferencing test step, only dcm, study, series and MRI weighs were given 
    - meaning I have to pick slice on my own
    - So pick samples based on given instance_number when train, but in test, pick equally with rule-based

- coordinates can be used as aux head?
    - maybe difficult because it is crop but may worth trying

- task for tmr
    - make seg_infer notebook inferencing on dicom file insetad of png
    - crop sagittal    

# 7.20

- segmentation using lumbar spine is not consistant as I thought
    - It would stretch too much when resized

- use coordinate instead maybe?
    - expand area from coordinate for cropping
    - expand to 64 w&h then resize to 320


- need to train seperately with sagittal and axial
    - re-adjust coordinate along with image size
        - new_coordinates = (new_image_size / original_image_size)

- segmentation may not need for coordinate?
    - use regression instead?

- let's try both start with segmentation
    - segmentation
        - RESULT
            - decent performance?
            - Do not need this model when training classification model
    - regression
        - RESULT
            - maybe I don't need this

# 7.21

- trained cooridinates segmentation model
    - an okay-ish performance

- task for tmr (What I need to do next?)
    - make notebook crop z-axis based on cooridinates
        - when train, pick sample evenly using quantile or linespace
    - axial can use center crop?

# 7.22

- check if I can crop all levels
    - to get all z-axis coordinates, I need to go through all inferenced image and pre-process them before crop voxel

- multi-label, multi-class, multi-input

# 7.23

- How can I handle picking image evenly
    - step
        - load each series for the study
        - predict/get coordinates, crop out 5 levels
        - each level have 3 channels and maybe 5 to 10 slices
            - start and end 3-offset to avoid unexposed spine
            - so shape would be like (bs, 15-30, img_size, img_size)
            - for ex.
                - if pick 5 samples evenly and then stacked from L1 to S1, (bs, 45, img_size, img_size)

- treats 1 study as 1 example
    - load 1 study, find all series and process the series based on series_description

- what needs to be done in preprocessing
    - predict/get coordinates
    - calculate mean coordinates

- task for tmr
    - handle incorrectly annoatated coordinates
        - used trained model to predict wrongly annotated coordinates
            - done
    - finish up training pipeline


# 7.24

- do I need to create 2.5d for every .dcm file?

- coordinates predictor did pretty good job
    - most of the coordinates are off like 1 to 2 pixels (max 3-4)

- handling axial
    - if I need to use axial, coordinates would be super difficult to use
    - What I can do is create model to predict coordinates, then draw manaul bbox based on that for cropping

# 7.25

- train only sagittal for now as axial plane need to handle a lot of things like missing file or consistency of the sequence

- have to write loop for handling incorrectly annotated coordindates
    - done
    - plus calculated bbox in create_dataset notebook

- Next, cropping axial and handling axial
    - use cross referencing then centercrop axial?

# 7.26

- picking slice
    - most series will doing find around 24 slices
    - then there is spike from there to around 50 slices
        - use `np.linspace` pick 20 samples
    
- feeding process and targets calculation
    - feed crop slices of level together with level labelled
    - predict muticlass of mutilabel either 
        - sagittal
            - T1
                - left_neural_foraminal_narrowing
                - right_neural_foraminal_narrowing
            - T2
                - spinal_canal_stenosis
        - axial
            - left_subarticular_stenosis
            - right_subarticular_stenosis
    - labels
        - head for each condition, each head contain 3 class for its severity
            - Sagittal
                - T1
                    - L_Narrow
                    - R_Narrow
                - T2
                    - SC_Stenosis
            - Axial
                - L_subbar
                - R_subbar


    - haven't doen anything with axial yet
        - so in current state will try training 9 labels of sagittal first

- task for tmr
    - need to rethink how I should feed image to model

# 7.27

- not sure if multi-head need

- so every batch output would be
    - ids: 
        - (STUYDY_ID)_(LEVEL_M)
    - imgs: only one level of certain study
        - N slices of sagittal T1 of level M
        - N slices of sagittal t2 of  level M
        - N slices of axial T2 <-*add later*
    - labels: 
        - 6 for sagittal T1
        - 3 for sagittal T2
        - 6 for axial T2 <-*add later*
        
        - 3 serverity for each condition
            - CONDITION_(normal/mild | moderate | severe)

- task for tmr
    - finish up creating label appropriate format
        - done
    - finish up training notebook
        

# 7.29

- training notebook almost finish, just have to handle missing file and other edge case

- handle missing file by replace it with np.zeros for now
    - can not use np.zeros because it can not resize
    - use previous or next .dcm is also doable

- training notebook ran
    - took roughly around 1 hour per epoch
    - this is only test run


# 7.30

- found bugged where every image got convert to np.uint8 before normalize

- need to look throught how I should pick slice
    - too much slices that do not help telling condition

- trying pick slices expanding from the middle index of all .dcm file list
    - now it 10 samples so -5 and +5 from the middle index

- multiple head option still viable


- log_loss from sklearn and CrossEntropyLoss is different
    - can not find out why

- task for tmr
    - looking more around loss calculation


# 7.31

- without: subar
- test_run2
    - diff
        - bugged image
    - log_loss (sus) : 0.23176
    - official_metric: 

- test_run3
    - diff
        - fixed image
        - cross_entropy
    - log_loss (sus) : 0.20504
    - official_metric: 

- test_run4
    - diff
        - bce in train
        - ce in valid
    - log_loss (sus) : 0.19504
    - official_metric: 


- integrate official metric to train pipeline?
    - done

- trying train without weight
    - both loss and score get so much better
    - nn.CrossEntropy's weight is only used for handling class imbalance not for severity punishing

- ideas for processing axial plane
    - center crop
        - exploring sizes for cropping
    - coordinates segmentation then draw bbox
        - need to check its boundaries
        - checking how coordinates are annoatated
    - find dataset for training segmentation then crop

- test_run5
    - implemented metric calculation incorrectly
        - both submission and solution need to to sorted beforehand
            - their row_id need to be the same before the calculation
    - need to calculate again separately
    - log_loss: 0.46949 from 2nd epoch (3rd epoch should have better score but the implement was incorrectly)

- task for tmr
    - work on axial plane as 

# 8.1

- it is possible that if study_id does not have sagittal T2 meaning that study also does not have axial T2

- there is one study that was not lumbar spine MRI scan remove later

- center crop is decent enough for a few first experiments
    - ex. if image size is 512x512 then image[128: 512 - 128, 128: 512 - 128] resulting 256 size of image, then resize down to align with other planes

- How can I pick the axial with the right level
    - axial T2 is matching with sagittal T2
    - Using proportion of sagittal coordinates-y to image size
        - ex.
            - y = 91.3213, image_size = 384
            - proportion to image is y/image_size = 91.3213/384 = 0.2378..
            - using this ratio in `np.percentile` or `np.quantile`
            - np.quantile(dcm_file_list, q=ratio).astype(int) will give appromately location of the level
            - then expand from that -/+ (start with -3/+3)

- in fold 0, the small image height is 192, larget image height is 960
    - using 256 center-crop would throw an error as some image is small than it become zero

- task for tmr
    - find a way better to center-crop
        - maybe use ratio instead of deterministic method 
    - create notebook that create .npy for faster training breaking into group might be necessary
        - done

# 8.2

- forgot to include axial evaluation in training loop
    - meaning that test_run6 ran without axial log_loss: 0.467606
    - but train&val loss is included 

- now use center-crop that utilize only to max size of image size
    - possibly prevent causing zero-size image
    - try with 224x224 first

- using .npy that normalized to range of 0 and 255 (np.uint8)
    - from around 1 hr/epoch to 13 mins/epoch
    - not sure why only 1 epoch took 13 mins but later epoch get faster (around 7 mins)
        - because of pin_memory?

- need to check dataset batch again to make sure image is feeding correctly

- room for improvement in early stage
    - re-train coordiantes segmentation model
    - add data augmentation
    - increase model size
    - hyperparameters
    - axial 
        - cropping method and size
        - slice position check
    - hyperparameters tuning

- using y-ratio to image size is only accurate to some case a lot of theme is inaccurate
    - need to find a better way

- task for tmr
    - find better way to matach axial t2 and sagittal t2
        - use ratio but this time pick base on patient position instead
            - comparing both position
        - picking the right level is might be the only last thing to do before driving into experiment

# 8.3

- all probabilities across severity does not have to be summed up to 1.0
    - meaning that Binary Cross Entropy is usable
    - try later, after done test running

- mutilabel stratified might be usable too?

- is sagittal in reverse order? left-right or right-left?

- in cross-referencing MRI planes notebook
    - sagittal_t2 z-axis (image position patient) is coordinates in real-world which reflect on top left corner of the sagittal t2 image (0, 0)
    - from there minus with pixel spacing attribute from dicom file as minus will moving toward toe (Axial in this compettion is scanned from Head to Toe)
    - minus until the end of the sagittal image size
    - not working
        - still inaccurate

- about previous method using ratio
    - if I can find where the axial scan start

- predicting all targets using only sagittal is also doable

# 8.4

- took a day off

- task for tmr
    - continue on working on locating axial
        - interval between slices in axial
        - adjusting by using start-middle-end index of files?
    - try bce

# 8.5

- what if I crop in voxel?
    - crop only 1 side (left/right) but include all slices
    - crop both sides by combining both bbox of coordinates
    - but still better if I can find the right level
    
- found outlier where instance jump from 24 to 5031 (series_id: 2683794967)

- found some kind of pattern in axial slices interval
    - if divide last instance_number in the series with 5, wiil get interval of the slices
        - comfirmed, there is slight diff -1/+1 in instance number but I can get interval number of the slices
            - thus, each interval will reveal each level of the vertebrae disc
        - found interval pattern the start and last of slices is still unknown

- next thing, find where l1_l2 start and its interval

- there are some series that in reverse order

- each condition can best diagnosis by
    - subarticular stenosis
        - axial t2
    - spinal canal stenosis
        - sagittal, axial t2
    - foraminal narrowing
        - sagittal, axial t2

- in the current state, feeding sagittal t1 and t2 make model predicting spinal canal is decent at certain level

- relationship between condition and location of the slices
    - spinal canal is usually at 0.5 quantile of all files (ST2)
    - left forminal is usually at 0.75 quantile of all files (ST1)
    - right forminal is uaully at 0.25 quantile of all files (ST1)
    - use this information to re-write feeder (data_preprocessing and dataset)
    - can do pick -1/+1 at each quantile
    - can other method I can do

- I should review Trigonometry

- task for tmr
    - review how the config pass down to the function especially number of image samples
    - thinking about threshold of the closest distance of axial t2
    - check out reverse order axial
        - done

# 8.6

- in reverse order series, there is FFS Patient position, (HFS were majority in training dataset)
    - Don't have to worry about that as minimum distance is calculated purely on coordinates, so order become less importnant


- about threshold of the minimum distance
    - seems like the exact match have lower than 0.
    - looking into slices thickness and spacing betwen slices might find the appropriate threshold
        - because min distance of each series represent differently 1 series might have great distance but only 1 slice diff but ohter series might be more 2 slices diff

- some time image is very difficult to recognize
    - maybe because of the coordinates?
    - check out later
    - maybe use coordinates predictor to predict coordinates only the middle idx of the series?

- need to rewrite train coor segmentation notebook
    - done
    - comeback as later will need to refine coordinates

- task for tmr
    - finish up inference notebook

# 8.7

- for coordinates of the level
    - asides from segmentation task, It can be replaced wiht regression task too
    - using coordinates as 5 targets with MSELoss

- using segmentation for predicting coordinates showed inconsistency where some levels can not be found then it get rejected

- need to write coord regression training notebook
    - train and test dataset class need to be created seprately
        - because the trainning set contain 1-3 set of 5 level indicate for each condition per set
    - filted out series that does not have 5 level

- maybe bbox window size need some rework like based on ratio rather than absolute pixel value like 64
    - because it will be resize anyway before feeding to model


- task for tmr 
    - almost finish inference notebook, finish it
        - what's done
            - coords predictions pipeline
            - sagittal t1 and t2 loading dataset
        - what's left
            - axial loading dataset
                - need patient's coord calculation
                    - take from sagittal t2
                    - then calculate min distance in axial


# 8.8

- finished infer notebook
    - still uncheck, need to check later


- prioritize training first as there is a lot GPU quota left

- **exp001**
    - [RESULT] fold0 CV: 0.5168, LB:
    - result from 3rd epoch and it start to overfitting from 4th epoch
        - maybe augmentation can help

- **exp002**
    - change crop to ratio of 0.2 instead of absoluate value 64x64
    - [RESULT] fold0 CV: 0.5166, LB:
    - still overfitting at late epoch
    - [KEEP]

- **exp003**
    - change crop to ratio of 0.2 to 0.125
    - [RESULT] fold0 CV: 0.5039, LB:
    - still overfitting at late epoch
    - [KEEP]

- **exp004**
    - change lr 3-e3 to 1e-3
    - [RESULT] fold0 CV: 0.5189, LB:
    - very unstable
    - [KEEP]

- **exp005**
    - change 5 to 10 epoch
    - [RESULT] fold0 CV: 0.5305, LB:
    - result from the 3rd epoch

- **exp006**
    - add augmentation
        - HorizontalFlip
        - RandomBrightContrast
        - ShiftScaleRotate
    - [RESULT] fold0 CV: 0.5177, LB:
    - result from the 3rd epoch

- **exp007**
    - add augmentation
        - HorizontalFlip
        - RandomBrightContrast
        - ShiftScaleRotate
        - Blur
        - Distortion
        - Dropout
    - [RESULT] fold0 CV: 0.5413, LB:
    - putiting augment too much? try taking one by one out later

- maybe redcue epoch to 3 from 5 might help?

- should I separate sagittal and axial?

- **exp008**
    - based: exp004
    - try taking axial out
    - [RESULT] fold0 CV: 0.5126, LB:

- task for tmr
    - look into which plane is the most effective for each labels


# 8.9

- **exp009**
    - use hard label instead of soft label
        - use class index instead of class probablity and add -100 to make loss ignore it
    - based: exp008
    - [RESULT] fold0 CV: 0.5218, LB:
    - not helping with overfitting

- **exp010**
    - based: exp008
    - change epoch from 5 to 3 and warm up from 0.1 to 0.375 (warmup only till the first epoch)
    - [RESULT] fold0 CV: , LB:
    
- **exp011**
    - based: exp008
    - change epoch from 5 to 3 
    - [RESULT] fold0 CV: 0.5351, LB:    

- **exp012**
    - based: exp008
    - change epoch from 5 to 3 
    - no warmup
    - [RESULT] fold0 CV: 0.5115, LB:
    - [KEEP]

- **exp013**
    - based: exp008
    - epoch 5
    - no warmup
    - [RESULT] fold0 CV: 0.5066, LB:
    
- **exp014**
    - based: exp013
    - epoch 5
    - no warmup
    - lr 3-e3
    - [RESULT] fold0 CV: 0.5008, LB:

- try out hyperparameters a bit then swith to effnetv2
- later try build my custom head

- **exp015**
    - based: exp013
    - epoch 5
    - no warmup
    - lr 5-e3
    - [RESULT] fold0 CV: 0.5042, LB:
    
- **exp016**
    - use efficientnetv2_rw_t.ra2_in1k
    - epoch 5
    - no warmup
    - lr 3-e3
    - [RESULT] fold0 CV: 0.5340, LB:

- **exp017**
    - use tf_efficientnet_b3.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3-e3
    - [RESULT] fold0 CV: 0.4877, LB:
    - b3 take only slightly longer
    - [KEEP]

- **exp018**
    - use tf_efficientnet_b3.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3-e3
    - bs 16 instead of 8
    - [RESULT] fold0 CV: 0.5406, LB:

- **exp019**
    - use tf_efficientnet_b3.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3-e3
    - img size 224 instead of 128
    - [RESULT] fold0 CV: 0.5210, LB:

- **exp020**
    - use tf_efficientnet_b3.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3-e3
    - img size 320 instead of 128
    - [RESULT] fold0 CV: 0.5089, LB:

- **exp021**
    - based: exp017
    - use efficientnetv2_rw_s.ra2_in1k
    - epoch 5
    - no warmup
    - lr 3-e3
    - [RESULT] fold0 CV: 0.5527, LB:

- **exp022**
    - based: exp017
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 5
    - no warmup
    - lr 3-e3
    - [RESULT] fold0 CV: 0.5249, LB:

- **exp023**
    - based: exp017
    - use convnext_tiny.in12k_ft_in1k
    - epoch 5
    - no warmup
    - lr 1e-4
    - [RESULT] fold0 CV: 1.0378, LB:
    - too low lr?

- **exp024**
    - based: exp017
    - use convnext_tiny.in12k_ft_in1k
    - epoch 5
    - no warmup
    - lr 1e-3
    - [RESULT] fold0 CV: 0.9976, LB:

- **exp025**
    - based: exp017
    - use tf_efficientnet_b5.ns_jft_in1k
    - epoch 5
    - no warmup
- lr 3e-3
    - [RESULT] fold0 CV: 0.5116, LB:

- **exp026**
    - based: exp017
    - use tf_efficientnet_b3.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - exp017 but 4 fold
    - [RESULT] fold0 CV: 0.4986, LB:
    - [RESULT] fold1 CV: 0.5240, LB:
    - [RESULT] fold2 CV: 0.5380, LB:
    - [RESULT] fold3 CV: 0.5118, LB:


- task for tmr
    - debug infer notebook check memory leak
        - done
        - turn pin_memory to False
    - look into each plane for each symptom
    - look into split group for stability
        - label every symptom to number as y
        - study_id_level as X
        - study as Group 
        - then split using StraitiedGroupKFold
    - exp026 is differentl from exp017 check that out
        - maybe it it better to split beforehand rather than split every time in test
        - if the result is still random then might be caused by polars somewhere or the model itself
        

# 8.10

- treats null as normal_mild
    - so I can label and split them consistantly
    - then fitler them out when score
    - need to create study that contained null value

- checking infernotebook
    - [x] coords
    - [x] feed cropped image


- do research for each symptom and their dianogsis
    - MRI types
    - T1-weighted MRI enhances the signal of the fatty tissue and suppresses the signal of the water.
    - T2-weighted MRI enhances the signal of the water.

- implement my own custom classifier head

- from exp027 using (v2)
    - fill null with normal_mild
    - label every symptom to number as y
    - study_id_level as X
    - study as Group 

- **exp027**
    - based: exp017
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - [RESULT] fold0 CV: 0.5198, LB:

- **exp028**
    - based: exp027
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - exp027 but 4 fold
        - checking its stability and reproducibility
    - [RESULT] 4 folds oof CV: 0.5056, LB:

- need to check left to right and rigth to left

- about picking slices
    - current pick 0.25, 0.5, 0.75 quantile of series array
    - could move a bit move toward middle like 0.3, 0.5, 0.7
    - if series contained 15 of slice, slice no. 5, 8, 11 are often the best visibility of spinal canal or forminal narrowing
    - I can pick slice base on ImagePositionPatient

- window_ratio_size can turn up a bit from 0.125 to 0.15 (0.2 is too much)

- **exp029**
    - based: exp017
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - window_ratio 0.15 instead of 0.125
    - [RESULT] fold0 CV: 0.5171, LB:
    - [KEEP]

- **exp030**
    - based: exp029
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - window_ratio 0.175 instead of 0.15
    - [RESULT] fold0 CV: 0.4993, LB:
    - [KEEP]

- **exp031**
    - based: exp030
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - window_ratio 0.2 instead of 0.175
    - [RESULT] fold0 CV: 0.5058, LB:

- **exp032**
    - based: exp030
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - intead of picking 0.25, 0.5, 0.75, pick 0.3, 0.5, 0.7 instead
    - [RESULT] fold0 CV: 0.5229, LB:

- move side of spine images toward mid did not help
    - maybe it get collapsed (reapeated) with mid 3 channels
    - or just simply missed most o fthe visible targets

- I think I can pick side of spine image by calculating distance of slices and pick a certain distance that would mostly show clear vision of targets

- **exp033**
    - based: exp030
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - included axial pick min_distance and its -1/+1
        - that make inputs up to 21 channels
        - thredshold: None
    - [RESULT] fold0 CV: 0.4978, LB:

- **exp034**
    - based: exp033
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - included axial pick min_distance and its -1/+1
        - that make inputs up to 21 channels
        - thredshold: 5.0
    - [RESULT] fold0 CV: 0.5031, LB:

- **exp035**
    - based: exp033
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - included axial pick min_distance and its -1/+1
        - that make inputs up to 21 channels
        - thredshold: 2.5
    - [RESULT] fold0 CV: 0.5052, LB:

- task for tmr
    - investigate distance calculation from middle to side of the spine
        - done
    - check scan direction whether the all scan goes the same diection (right to left or left to right)
        - done
    - think about separate axial to another model
        - note sure about this yet
    - implement my own custom classifier head

# 8.11

- https://blog.redbrickai.com/blog-posts/introduction-to-dicom-coordinate

    > The positive X direction is towards the left of the patient. (right to left)
    > The positive Y direction is towards the back of the patient (anterior to posterior).  
    > The positive Z direction is towards top of the patient (inferior to superior).  

- sagittal direciton: check by condition columns and its given instance_number

- submission still threw error
    - need to debug later
    - what can be possible errors?

- slice location is shared between sagittal t1 and t2, IPP is slightly off

- range from middle of the spine scan is likely to be 16 to 18 mm
    - can calculate by slicelocation and spacing between slices
    - calculate min distance by 

- from exp036, fold split might be a bit different (v3)

- **exp036**
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - excluded axial for testing
    - [RESULT] fold0 CV: 0.5052, LB:

- There is potential that is getting weird because there are left to rigth and right to left series

- If the x-coordinate increases, slices are moving from left to right.
- If the x-coordinate decreases, slices are moving from right to left.

- from exp037 using v4
 
- fixed left to right mixed up
    - pick slice based on min distance and then compare direction of IPP and slice location
    - if both goes the same way keep it, else swap left and right

- **exp037**
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - fixed left and right mixed up 
    - left to right is now align the same across all dataset
        - this should help forminal narrowing category
    - [RESULT] fold0 CV: 0.5265, LB:

- can not use only t1 or only t2 it will cause nan loss

- from exp038 using v5

- **exp038**
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - no warmup
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - [RESULT] fold0 CV: 0.4997, LB:


- **exp039**
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - [RESULT] fold0 CV: 0.4764, LB:

- **exp040**
    - exp039 but 4folds
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - [RESULT] 4 folds oof CV: 0.52188, LB: 
        - model become so random
        - need to address this

- task for tmr
    - implement my own custom head 
    - try out convnext
    - try stack image, merge them into one side
        - can not because coordinate is pointing to the same objective
    - try pick base on quantile but decide left-right with IPP

# 8.12

- before swith model, I need to make custom head first else it will give unstable result
    - seed_everything right before get_model seems to fixed the initilized weight in classifier head

- **exp041**
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - try input 9 channels
        - first 3 chans (left)   is sagittal t1
        - next  3 chans (middle) is sagittal t2
        - last  3 chans (right)  is sagittal t1 
    - [RESULT] fold0 CV: 0.5036, LB:

- **exp042**
    - based: exp041
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - added axial t2
    - [RESULT] fold0 CV: 0.5362, LB:

- **exp043**
    - exp040 fold0
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use only sagittal and remove subarticular
    - [RESULT] fold0 CV: 0.4197, LB:


- might have to seperate 2 model
    - sagittal and axial
    - sagittal feed model by level
    - axial feed by series (include all level)

- axial right or left may can determine by sagittal t2

- for level locating 
    - can use sagittal t2 cross-ref or 
    - each disc distance is about 24-30 mm in IPP and SliceLocation
        - the problem is starting point

- just found potential bug
    - I have not check correctness of select axial t2 in train loop
    - may have implemented incorrectly?

- in axial plane, there is diffrent order exist (head to toe, toe to head)

- task for tmr
    - work train subarticular notebook
        - facing nan loss, maybe there are some study_id does not have axial plane
            - check on that

# 8.13

- fixed nan looss by removing series theat does not have 5 level and unaligning coordinates of patient and IPP
    - but still not achieving good result, my guess it that images is not picked that at the right image
    - need to try different strategy

- **exp044**
    - based: exp043
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use only sagittal and remove subarticular
    - group left, mid, right
    - [RESULT] fold0 CV: 0.4134, LB:

- **exp045**
    - based: exp044
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - group left, mid, right
    - [RESULT] fold0 CV: 0.4797, LB:


- **exp046**
    - exp045 but 4 folds
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - group left, mid, right
    - [RESULT] 4 fold oof CV: 0.530, LB:

- **exp047**
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - *use 5 head separately for each symptom*
    - group left, mid, right
    - [RESULT] fold0 CV: 0.4838, LB:
    - [RESULT] fold1 CV: 0.4787, LB:
        - since each head is indenpendent from others, they won't be influence by others result
        - spinal and foraminal's classification is improve without bounce back like single-head
        - however, subarticular is still difficult to train, though it is also improving
    - [KEEP]

- split is very unstable

- task for tmr
    - look into split fold
        - use MultilabelStratified?
    - can expand middle channels help subarticular?

# 8.14

- multilabel is doable try later after a few experiements over other stuff like diff model, expanding mid channels

- **exp048**
    - *exp047 but 4 folds*
        - just for testing stabibility and reproducibility of the training loop
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - [RESULT] 4 folds oof CV: 0.5028, LB:
        - reproduced exp047 both fold 1 and fold 0

- **exp049**
    - based: exp047 
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - *use max pooling*
    - [RESULT] fold0 CV: 0.4789, LB:

- **exp050**
    - based: exp047 
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - *use gem pooling*
    - [RESULT] fold0 CV: 0.4867, LB:

- **exp051**
    - based: exp047 
    - *use tf_efficientnetv2_s.in21k_ft_in1k*
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - [RESULT] fold0 CV: 0.4928, LB:

- **exp052**
    - based: exp047 
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - *expanding middle (spinal) from 3 to 5*
    - [RESULT] fold0 CV: 0.4670, LB:
        - expanding help a lot in subarticular
    - [KEEP]

- current facing problems and tasks
    - overfitting at the late epoch
    - which augmentation to use

- **exp053**
    - based: exp047 
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - *not grouping left, mid, right*
        - sagittal t1 then sagittal t2
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.4761, LB:

- **exp054**
    - based: exp052
    - *use tf_efficientnetv2_s.in21k_ft_in1k*
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - *change lr from 3e-3 to 3e-4*
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.5506, LB:

- **exp055**
    - based: exp052
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - *lr 1e-3*
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.4761, LB:
        - valid loss: 0.412949

- **exp056**
    - based: exp052
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - *lr 4e-3*
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.4750, LB:
        - valid loss: 0.403689

- **exp057**
    - based: exp052
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - *lr 2e-3*
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.4760, LB:
        - valid loss: 0.404563

- **exp058**
    - based: exp047 
    - use tf_efficientnet_b3.ns_jft_in1k
    - epoch 5
    - warm up 1st epoch (warmup ratio 0.2)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.4712, LB:


- task for tmr
    - working on regularization (reduce overfitting)
        - image augmentation
        - dropout
        - train longer?
        - increase batch size?
    - if inference notebook threw error, then debug
        - memory leak
            - read only first and last then calculate all slice with spacing based on path of instance number

# 8.16

- debugging inference notebook
    - have not found success in submission

- pydicom is causing memory leak
    - switch to dicomsbl and memory is almost not pling up like pydicom did

- currently, using calculation from firsw and the last slice image position patient
    - it saved a lot of retrieving time and memory
    - in most case would have slight error, but a few have a decent amount of error from the real tag

- try submit with above changed tmr, if it is still caused error then try switch back to retrieve the real tag

- **exp059**
    - based: exp047
    - use tf_efficientnet_b0.ns_jft_in1k
    - *epoch 10 instead of 5*
    - *warm up until 2nd epoch (warmup ratio 0.2)* (forgot to change)
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.4767, LB:
    - [KEEP]

- **exp060**
    - based: exp047
    - use tf_efficientnet_b0.ns_jft_in1k
    - *epoch 10 instead of 5*
    - *warm up until 1st epoch (warmup ratio 0.1)*
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - [RESULT] fold0 CV: 0.4899, LB:

- task for tmr
    - submit newly applied dicomsdl and some new handlings
        - this should be worry free of kernel dead
    - start trying image augmentation
    - exploring regularization techniques like mixup, dropout

# 8.17

- submitted new adjusted inference notebook but still failed
    - currently, infer_data_preprocess is the most suspicious


- **exp061**
    - based: exp059
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - HoriFlip
        - VertFlip
        - ShiftScaleRotate
    - [RESULT] fold0 CV: 0.4700, LB:
    - [KEEP]

- **exp062**
    - based: exp059
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4651, LB:
    - [KEEP]

- **exp063**
    - based: exp059
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - RandomBrightnessContrast
        - Blur
        - Distortion
    - [RESULT] fold0 CV: 0.4642, LB:
    - [KEEP]

- **exp064**
    - based: exp059
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - HoriFlip
        - VertFlip
        - ShiftScaleRotate
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4700, LB:

- **exp065**
    - based: exp059
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - HoriFlip
        - VertFlip
        - ShiftScaleRotate
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.5134, LB:
        - need longer training time?

- **exp066**
    - based: exp059
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4589, LB:

- **exp067**
    - based: exp065
    - use tf_efficientnet_b0.ns_jft_in1k
    - *epoch 15 instead of 10*
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - HoriFlip
        - VertFlip
        - ShiftScaleRotate
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4920, LB:
        - training loss is still high and barely reduce at late epoch

- **exp068**
    - based: exp066
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - ShiftScaleRotate
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4604, LB:

- **exp069**
    - based: exp066
    - use tf_efficientnet_b0.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - HorizontalFlip
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4666, LB:

- seems like sptial-level transforms is advert to the result
    - symptom is always the same side?
        - is there a chance that MRI is scanned while patient facing down?

- task for tmr
    - submit inference notebook, check, debug

# 8.18

- turned out the error was caused in predict_cv function
    - potenttially in dataset class
    - which could come from inter_data_preprocess

- **exp070**
    - based: exp066
    - use tf_efficientnet_b3.ns_jft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4738, LB:

- **exp071**
    - based: exp066
    - use tf_efficientnetv2_s
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4842, LB:

- **exp072**
    - based: exp066
    - use tf_efficientnetv2_s
    - epoch 10
    - warm up ratio 0.2
    - *lr 3e-3 to 3-e4*
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *add augment*
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.5031, LB:

- **exp073**
    - based: exp066
    - *use tf_efficientnet_b4.ns_jft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.47021, LB:

- **exp074**
    - based: exp066
    - *use tf_efficientnet_b1.ns_jft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4661, LB:


- task for tmr
    - continue on debug inference notebook
    - exploring 2.5d with RNN head

# 8.19

- inference notebook still raise error
    - but found that it is found dataset

- take a day off
    

- task for tmr
    - submit infer notebook with try/catch the whole `__getitem__` exception as all zeros
    
# 8.20

- finally found bug
    - the bug is in these steps
        - read dcm pixel array
            - if anything
                - the file not found (high)
                - the file does not have pixel array
        - slice the ROI
            - almost impossible to occur any errors
        - cv2.resize
            - handled by clipped max image size and zero
        - replace image_stack channel
            - the error that encountered was trying to replace out of the index

- **exp075**
    - based: exp066
        - but 4 folds
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 fold oof CV: 0.4999, LB: 0.54


- **exp076**
    - based: exp066
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *added axial pick based on SliceLocation*
        - increase to 25 channels
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4701, LB:

- maybe use axial is not very good idea for as it worsen when including in training
    - I should focus on picking the right slice for the time being

- what I can improve?
    - try to crop only one level
    - right there is other level shown in the image

- task for tmr
    - exploring and re-check EDA cropping level
        - including padding to main its ratio

# 8.21

- **exp077**
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *cropped level manually to cover only 1 level*
        - *then resize to longest max size 128 and padded make it become square*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4775, LB:
    - [RESULT] fold3 CV: 0.5309, LB:


- Some analasis from exp075
    - level l4_l5 doing the worse out of others
        - level l4_l5 tends to have more severe sample, while others is normal or moderate dominating


- task for tmr
    - do more analysis and decide what to do next to address current situation like
        - how many slices should be used
        - how to implement RNN head properly
        - how each symptoms is diagnosis

# 8.22

- axu loss
    - segmentation head
        - to create mask, train only on visiable?

- if I use model to find axial level
    - I can train model that targets is level and feed image to model randomly
        - labels are [spine, l1_l2, l2_l3, l3_l4, l4_l5, l5_s1]
    - doubt that it would be more accurate than using IPP or slicelocation
    

- why exp077 does not show any improvment?
    - the new manual is more visible than using window ratios
        - the window ratios is somewhat has a better resolution because its depends on image size
        - however, it is also sometimes show others level

- **exp078**
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - *window ratio 0.12 instead of 0.175*
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4749, LB:
    - [RESULT] fold3 CV: 0.5045, LB:
    - [KEEP]?
        - I have feeling that the previous window ratio (0.175) is caused overfitting
        - but I do not have any evidents

- Sagittal T2 sides is not very visiable, show I remove it?

- **exp079**
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *take 3 slice side of spine of sagittal t2 out*
        - both left and right
        - that make channel from 22 to 16
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4729, LB:
    - [RESULT] fold3 CV: 0.4932, LB:
    - [KEEP]

- **exp080**
    - use tf_efficientnet_b0.ns_jft_in2k
    - *epoch 15 instead of 10*
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - take 3 slice side of spine of sagittal t2 out
        - both left and right
        - that make channel from 22 to 16
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4888, LB:

- task for tmr
    - do more analysis and decide what to do next to address current situation like
        - how many slices should be used
            - partly done
                - by taking out sides of spine in Sagittal T2 series
            - have not prove yet
                - have to train all 4 folds then submit to validate the idea
            - have a feeling that there is a better way to calculate min distance accurately
                - in merged train df, the coordinates is not on the same slices. So surely there is a pattern for each level
        - how to implement RNN head properly
            - after done with slices at certain level, exploring this
        - how each symptoms is diagnosis
            - leave this topic after done above for further improvement in detail.

# 8.23

- found bug? and fixed
    - the slices is sorted based on their direction


- **exp081**
    - based: exp079
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - take 3 slice side of spine of sagittal t2 out
        - both left and right
        - that make channel from 22 to 16
    - *sorted slices to their direction*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4824, LB:

- cropped level can shift a bit more to left?
    - need to be careful not to go over foraminal

- **exp082**
    - based: exp079
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - *use focal loss*
        - *gamma 2*
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - take 3 slice side of spine of sagittal t2 out
        - both left and right
        - that make channel from 22 to 16
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4803, LB:

- task for tmr
    - train 4 folds of exp079
    - try out LSTM head

# 8.24

- **exp083**
    - based: exp079
        - *4 folds*
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - take 3 slice side of spine of sagittal t2 out
        - both left and right
        - that make channel from 22 to 16
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4729, LB:
    - [RESULT] fold1 CV: 0.4670, LB:
    - [RESULT] fold2 CV: 0.5161, LB:
    - [RESULT] fold3 CV: 0.4932, LB:
    - [RESULT] 4 folds oof CV: 0.4876, LB: 0.60

- **exp084**
    - based: exp079
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - *each head recieve extracted features from LSTM head*
        - *then take mean of seq as logits*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - take 3 slice side of spine of sagittal t2 out
        - both left and right
        - that make channel from 22 to 16
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4643, LB:

- **exp085**
    - based: exp079
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - *each head recieve extracted features from LSTM head*
        - *then take mean of seq as logits*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *put back 3 slices from sagitta T2*
        - that make it back 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4477, LB:
    - [KEEP]

- **exp086**
    - based: exp079
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - *2 LSTM layers instead of 1*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4774, LB:

- **exp087**
    - based: exp079
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits 0.45
        - *logits is now modified to have dropout*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4470, LB: 1.07
    - [KEEP?]
    - overftting if submit only one fold

# 8.25

- **exp088**
    - based: exp079
    - use tf_efficientnet_b0.ns_jft_in2k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - *instead of normalization, divide by 255 instead*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4672, LB:

- What I don't understand right now
    - why sorting slices to go exactly as scan given worse result
        - have not tried thoroughly
    - why lower CV with fewer slices (22 -> 16) to worse in LB
        - Maybe doing error analysis can help
        - need to come up with better solution than confusion matrix

- What have not tried yet
    - MixUp
    - as latest best improvement did not shown obvious ovefitting in loss
        - need to address this issue by
            - adjusting lr
                - lower lr seems to be the way
            - increase epoch
            - scaling up to slightly bigger model
    
- from exp089
    - testing out effnetv2
        - starting with current lr (3e-3)
        - then try decrease slighly iteratively

- **exp089**
    - based: exp079
    - *use tf_efficientnetv2_s.in21k_ft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-3
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.5018, LB:
    - took 2 hrs to train per fold

- **exp090**
    - based: exp087
    - *use tf_efficientnetv2_s.in21k_ft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - *change lr from 3e-3 to 3e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4212, LB: 0.47
    - [KEEP]

- finding optimal distance from middle spine slices to foraminal slices
    - currently, the most fit range is from 16mm~18mm
    - surely there is a pattern
    - so from the center (middle) the sides around 3 or 4 slices from the mid
        - currently, method is also picking the same slice
        - I can implement this in inference notebook where IPP does not exist and need to be picked manually
            - done

- try 1 head later, as 5 head might cause overfitting
    - or same LSTM but diffrent Linear heads
    - or same LSTM but diffrenet Linear heads with specific seq

- need to look deeply into nature of log loss


- **exp091**
    - based: exp087
    - *use tf_efficientnetv2_s.in21k_ft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
            - *now only 1 LSTM layer and pass to each Linear head*
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4259, LB:

- **exp092**
    - based: exp087
    - *use tf_efficientnetv2_s.in21k_ft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
            - *now only 1 LSTM layer and pass to only 1 Linear head*
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4243, LB: 0.48

- task for tmr
    - check inference version if why there are big gap
    - submit exp092


# 8.26

- found bug in inference notebook
    - when change model to LSTM head, change n_slices as seq was forgotten
    - fixed

- I can write coordinates error handling
    - I can take mean of each series and each level then replace it when the value is nan or null even zero

- more note about coordinates
    - the predicted coordinate is less consistent than the given data
    - so there is a big chance that the result is showing big gap because of this 
        - saying that the coordinates might point to a bit off the money

- I have to handle OpenCV possible error
    - because the dicom file is already read once when create IPP df
    - meaning that the error is coming from cv2.resize
    - currently, handling by clip coordinates to min of 1 and max of mean img size

- lower level of lumber spine is a bit wider than upper level
    - meawning that the coordinates/diagnosis are sometimes point the slight outer of the series

- aligning T1 and T2 might help model predict a better result?
- check up again or left and right of sagittal
    - what I want is left on early order and right on late order
- exapanding side of spine to outer/inner 1 more slice
    - it may depend on direction of series

- currently, gap between CV and LB is about 0.05

- so multi-head is better

- **exp093**
    - based: exp090
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - *mean took from that certain range of seq responding to the slice location*
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4301, LB: 

- **exp094**
    - based: exp087
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *change lr from 3e-4 to 3e-5*
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4322, LB: 

- **exp095**
    - based: exp087
    - *use convnextv2_pico.fcmae_ft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - *lr 3e-5*
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4351, LB: 

- task for tmr
    - read discussion, looking for potential improvement

# 8.27

- what I can do to further improve the result
    - augmentation
        - Mixup
        - HorizontalFlip
            - the previous exp was underfitting
        - Hue

    - thresholding result
        - lower lumber spine tends to have more moderate and severe labels
    
    - integrate Axial
        - I had already find to way to locate them
        - the problem is where feed them to network

    - auxiliary head
        - add head to predict level?
            - this might help network understand specific level
            - then, it might give a certain bias toward certain level
            - like upper level tends to have less such symptom than lower level
    - RNN head
        - GRU
            - not sure if it better than LSTM

    - image size
        - image size after cropped
            - currently 128
            - can go up to 224
        - window size ratio
            - currently 0.12
            - can go up more a bit

    - cropping
        - different aspect for each level
            - l1 to l3 can be the same
            - l4 to s1 might need to tilt a bit
    
    - model architecture
        - maxvit
            - have not train yet
            - might need max norm as it is based on transformer   
        - convnextv2
            - trained pico size model
                - decent performance
                - it needed small lr than effnet
                - took around 13 mins/epoch

            - bigger might yield better result
        - convnext
            - have not train yet
            - not sure yet is it better than v2

- **exp096**
    - based: exp090
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - *add aux head predicting level*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4262, LB: 0.49
    - looking at result, feeling this have good potential if tuning at the right hparams
    - [RESULT] fold0 re1 CV: 0.4110, LB: 0.47
        - this exp +1 on argmin

- check out idx of argmin
    - stick with + 1 for a while
    - exp090 (current best) is not + 1

- currently applied to train pipeline that can be removed later (diff from exp090)
    - aux head

- task for tmr
    - run test on colab?
    - thinking about potential of multilabel

# 8.28
- colab unzipping credit consumption
    - start: 103.79
    - unzip end: 103.5
    - finished: 100.86
    - took around 2-3 credits per run
    - note that colab ran a bit diff from kaggle

- **exp097**
    - based: exp090
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - change lr from 3e-3 to 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *adjusted idx by + 1 to argmin*
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4187, LB: 0.49
        - valid loss: 0.3570
    - [RESULT] fold0 re CV: 0.4313, LB: 0.47
        - valid loss: 0.3417
    - this is weird and I feel like I missed something

- seems like there is a trend that lesser loss give better score rather than official metric calculator

- task for tmr
    - submit exp097_re which this time best valid loss as metric

# 8.29

- after submitted expo97_re 
    - this one picked based on lowest valid loss, though the log loss score is worse
    - lower valid loss gained better log loss score marginally
    - however, log loss is also need to be taken into account as better valid loss but worse log loss does not give better LB
    - It has to get better on both loss to gain better LB result
    - For example
        - exp090 and exp097 is almost the same except side spine selection where exp097 is + 1 to argmin idx
        - exp090
            - VL: 0.3453, CV: 0.4212, LB: 0.47
            - avg of both loss: 0.38325
        - exp097
            - VL: 0.3570, CV: 0.4187, LB: 0.49
            - avg of both loss: 0.38785
        - exp097_re
            - VL: 0.3417, CV: 0.4313, LB: 0.47 (lower or equal to exp090)
            - avg of both loss: 0.38650

- How can I prove which one is better?
    - train 4 folds of both version?


- **exp098**
    - based: exp097
        - 4 folds
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *adjusted idx by + 1 to argmin*
    - *save based on both loss*
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 folds oof (valid loss) CV: 0.4444, LB: 0.44
        - fold0 VL: 0.3417, CV: 0.4313
        - fold1 VL: 0.3588, CV: 0.4291
        - fold2 VL: 0.3759, CV: 0.4469
        - fold3 VL: 0.3709, CV: 0.4691
    - [RESULT] 4 folds oof (log loss)   CV: 0.4413, LB: 0.45
        - fold0 VL: 0.3570, CV: 0.4187 *diif here*
        - fold1 VL: 0.3588, CV: 0.4291
        - fold2 VL: 0.3759, CV: 0.4469
        - fold3 VL: 0.3709, CV: 0.4691
    - colab creadit
        - start: 91.92
        - end upzip: 91.73
        - end fold0: 88.8
        - end train: 80.29

- task for tmr
    - submit exp098 best log loss models
        - done

# 8.30

- **exp099**
    - based: exp090
        - 4 folds
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *not adjusting idx to argmin*
    - *save based on both loss*
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 folds oof (valid loss) CV: 0.4642, LB: 0.46
    - [RESULT] 4 folds oof (log loss)   CV: 0.4483, LB: 
    - colab creadit
        - start: 80.29
        - end upzip: 80.08
        - end train: 69.96

- task for tmr
    - don't forget to copy code from colab to replace in kaggle
        - done
    - next thing to try
        - revisit aux head
        - try out multilabel
            - maybe not necessary
        - GRU instead of LSTM
            - not necessary, a bit worse than LSTM
        - adjust augment or add more diff augment
        - mixup
        - usage of axial

# 8.31

- **exp100**
    - *re-run exp098 fold0 for future reference/comparision*
    - based: exp098
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4322, LB: 0.46S (picked valid loss)

- **exp101**
    - based: exp100
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - *each head recieve extracted features from GRU head*
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 22 slices
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4320, LB: 

- **exp102**
    - based: exp100
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *25 slices*
        - *add 3 axial t2*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4146, LB: 0.47
    - [RESULT] fold2 CV: 0.4467, LB: 0.47

- task for tmr
    - try submit fold0 exp098

# 9.1

- **exp103**
    - based: exp102
    - *use tf_efficientnet_b0.ns_jft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *25 slices*
        - *add 3 axial t2*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold2 CV: 0.4545, LB:
        - too almost the same time as effnetv2


- it gets very confusing at this point
    - there is ddisprency between colab notebook and kaggle notebook
        - might cause by different gpu and torch version
    - include axial t2 seems to have relatively positive resultA
        - currently, exp102_f0 (included axial t2), is a bit better than exp098_f0
            - according to the precision sorting
        - can't say that now, have to submit exp102 fold 2 to see result first
        - plus CV strategy is StratifiedGroupKFold, each fold is have significant different to each others
        - best way to compare is to train all folds but it is very expensive to train


- task for tmr
    - submit exp102 fold 2
        - done
    - train 4 fold of exp102 on colab
        - can not be so sure, it would improve
        - need to further experiment
    - try decease lr to 2e-5 (try this after see results from 2 above)
        - or revisit aux head predicting level
        - or maybe add more augments

# 9.2

- **exp104**
    - based: exp102
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
        - *added aux head predicting level*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *25 slices*
        - *add 3 axial t2*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4219, LB:

- **exp105**
    - based: exp102
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 3e-4
    - fixed left and right mixed up 
    - *window ratio 0.12 to 0.15*
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *25 slices*
        - *add 3 axial t2*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4231, LB:

- **exp106**
    - based: exp102
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 3e-4 to 2e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *25 slices*
        - *add 3 axial t2*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4213, LB:

- **exp107**
    - based: exp102
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 3e-4 to 4e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - *25 slices*
        - *add 3 axial t2*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4164, LB: 0.46 (better than exp102)

- **exp108**
    - based: exp107
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - *increase image size from 128 to 224*
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4143, LB: 0.46 (better than exp107)
        - so it is certain that bigger image is better by margin
        - use this size in the final sub

# 9.3

- **exp109**
    - based: exp107
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4146, LB: 

- left to right or right to left is not addressed on axial T2
     - that might confused left or right for model

- In axial T2
    - HFS, left is on the right hand side of image while right is its encounter
    - FFS, left is on the left hand side of image and same with the right

- kind of ran of idea

- task for tmr
    - exploring what I can do to improve


# 9.4

- after assessing score of each level
    - lower level is harder to predict than upper like L1-L3
    - for L4-S1 show higer error (log loss)

- for better cropping axial
    - train model that predict x and y in axial instead of center crop

- **exp110**
    - based: exp107
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - *use coords as center when crop*
            - *window ratio size 0.35*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4145, LB: 0.47

- task for tmr
    - It is pretty obvious that the problem is at L4_L5
        - address this issue will dramatically improve the score
    - Do EDA on labels on each symptom and level

# 9.5

- per EDA
    - Majority of the severe case is grouping in lower level like l4_l5 followed by l5_s1

- **exp111**
    - based: exp110
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - *use 2 backbone*
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - *use coords as center when crop*
            - *window ratio size 0.35*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4654, LB: 

- **exp112**
    - based: exp110
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - *window ratio 0.15*
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - *use coords as center when crop*
            - *window ratio size 0.35*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4148, LB: 0.48

- task for tmr
    - finish exp112 record
    - explore options for improvement in the past competiions

# 9.6

- **exp113**
    - based: exp110
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - *middle (spinal) from 5 to 3*
    - *turn model into 3 in_chans*
    - *21 slices*
        - each 9 from sagittal t1 and t2
        - 3 axial t2
            - *use coords as center when crop*
            - *window ratio size 0.35*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4444, LB:

- **exp114**
    - based: exp110
        - *4 folds*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - *use coords as center when crop*
            - *window ratio size 0.35*
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 folds oof CV: 0.4462, LB: 0.44 (VL)
    - [RESULT] 4 folds oof CV: 0.4427, LB: 0.43 (LL)


- task for tmr
    - suspect that fold 3 is might be similar to test set


# 9.7

- submit exp114 fold 3 if submit quota 1 left

- **exp115**
    - based: exp114
    - *use coatnet_nano_rw_224.sw_in1k*
    - epoch 10
    - warm up ratio 0.2
    - *lr 4e-5*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4488, LB:

- **exp116**
    - based: exp114
    - *use maxvit_tiny_tf_512.in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - then take mean of seq as logits
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4493, LB:

- task for tmr
    - explore a bit more about diff arch
    - look more for diff augment

# 9.8

- using segmentataion head as aux head option still viable but have not implemented yet
    - but is not cropping already done a job for telling model to pay extra attetion to specific part of image?

- have not much tweaking Augmentation much yet

- **exp117**
    - based: exp114
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - *then take mean of seq as logits <- no longer need*
        - *added MLPAttentionNetwork*
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4272, LB: 0.46
        - quite low on valid loss 

- **exp118**
    - based: exp114
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - *then take mean of seq as logits <- no longer need*
        - *added Attention*
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4143, LB: 0.45 (better than exp114_f0)
        - valid_loss: 0.3340
    - [RESULT-RE] fold0 CV: 0.4091, LB: 0.47
        - valid_loss: 0.3372

- task for tmr
    - check each fold for each level of OOF
    - explore a bit more about how head should looks like
    - look more for diff augment

# 9.9

- checkout new model hgnet
    - mode seems to have small size yet quite impressive performance

- **exp119**
    - based: exp114
    - *use hgnetv2_b4.ssld_stage2_ft_in1k*
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: , LB:
        - did not finish training, as it does not look good in early epoch


- **exp120**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *warm up ratio reduced to 0.1 from 0.2*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4126, LB:

- **exp121**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4264, LB:

- **exp122**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - *Removed Blur*
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4226, LB:

- it was not the blur that ruin image but it was GaussNoise

- task for tmr
    - change again with fixed gaussnoise
    - adjust CoarseDropout
    - adjust distortion
    - check out mixup

# 9.10

- **exp123**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
            - *adjust gaussnoise range from 5,9 to 0.0,0.1*
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4196, LB:
    - overfitting started at 8th epoch
        - my guess is it need heavy augmentation espcially noise


- **exp124**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
            - *adjjust gaussnoise range from 5,9 to 0.1,0.4*
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4244, LB:

- MixUp seems computationally expensive to train
    - as final model that will be using image size 224 is already took so much time I can't afford this to run multiple experiment

- next thing to try
    - ShiftRotateScale is worth trying
    - increase cutout size ratio 0.2 from 0.1 and reduce number of box to 2-3 instead 4

- How can I use mask
    - segmentation aux head

- **exp125**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
            - *increase cutout size ratio 0.2 from 0.1 and reduce number of box to 2 instead 4*
    - [RESULT] fold0 CV: 0.4166, LB: (LL)
    - [RESULT] fold0 CV: 0.4246, LB: (VL)

- task for tmr
    - submit exp118_re
    - try ShiftRotateScale 

# 9.11

- **exp126**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
        - *instead of normalized in study_id_lvel, simply divide by 255.*
        - *might help consevere the most information in image*
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: , LB: 
        - did not finish training

- **exp127**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - *full precision training*
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.3986, LB: 0.46

- **exp128**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - *epoch from 10 to 15*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.3987, LB: 0.46

- maybe increase learning rate a bit might help like 5e-4

# 9.12

- **exp129**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - *epoch from 10 to 20*
    - *warm up ratio from 0.2 to 0.1*
    - *lr from 4e-4 to 5e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4065, LB: 

- **exp130**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4125, LB:
    - looking good
        - training loss still a lot bigger than validation loss
        - may have to adjust epoch or learning rate   
        - start with learning rate
            - 4-e5?

# 9.13

- **exp131**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - *epoch from 10 to 15*
    - *warm up ratio from 0.2 to 0.66 (1st epoch)*
    - *lr from 4e-4 to 4e-5*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4166, LB:

- **exp132**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - epoch 10
    - *warm up ratio from 0.2 to 0.1 (1st epoch)*
    - *lr 4e-4 to 3e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4116, LB:

- **exp133**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - epoch 10
    - *warm up ratio from 0.2 to 0.1 (1st epoch)*
    - *lr 4e-4 to 2e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4215, LB:

- task for tmr
    - train exp130 but warmup 0.1 and 20 epoch

# 9.14

- **exp134**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - *epoch from 10 to 20*
    - *warm up ratio from 0.2 to 0.05 (1st epoch)*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4227, LB:

- **exp135**
    - based: exp127
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - epoch 10
    - *warm up ratio from 0.2 to 0.1 (1st epoch)*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4178, LB:

- **exp136**
    - based: exp127
        - *re-run on kaggle kernal*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - full precision training
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4185, LB:

- **exp137**
    - based: exp127
    - *use tf_efficientnet_b4.ns_jft_in1k*
    - full precision training
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4326, LB:

- what to do next
    - try cutmix
    - try mean, max from RNN head
    - exploring more in the past compeition

# 9.15

- before moving on, I want to check out GaussNoise again first

- **exp138**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
            - var_limit 0.1 to 0.2
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4249, LB:

- **exp139**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
        - *CutMix*
    - [RESULT] fold0 CV: 0.4160, LB:

- **exp140**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - *max norm 1.0*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - added Attention
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4197, LB:

- add lstm mean and max head as ensembles
- maybe add maxvit or convnext later?

- **exp141**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - *try lstm mean and max head*
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4144, LB: 0.46

# 9.16

- **exp142**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *warm up ratio from 0.2 to 0.1*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - *removed Noise*
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4016, LB: 

- **exp143**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *warm up ratio from 0.2 to 0.0 (no warmup)*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - *removed Noise*
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4146, LB: 

- seperate upper and lower level on spine

- **exp144**
    - based: exp142
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.1
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - *train only 3 upper levels*
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - MixUp
    - [RESULT] fold0 CV: , LB: 

- **exp144-2**
    - based: exp142
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.1
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - *train only 2 lower levels*
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - MixUp
    - [RESULT] fold0 CV: , LB: 

- seperating not really help

- what to do next?
    - maybe I should start working on ensembling and post-processing

# 9.17

- last submission ensembles
    - effnetv2_s (img_size 224 or 128?)
        - lstm_attn head
        - lstm_mean_max head
    - convnextv2_tiny or nano
        - lstm_attn head
        - lstm_mean_max head
    - maxvit 224? (not sure yet)

- taking a day off

- task for tmr
    - train exp142 4 folds
        - both heads
    - try out maxvit or convnextv2 again
        - watch out for learning rate

# 9.18

- I want to check out diff on image size between 128 and 224
    - by exp145 will be trained on 128
    - and exp146 is on 224

- **exp145**
    - based: exp142
        - *setup and config is exact the same except ran on colab*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *warm up ratio 0.1*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4216, LB: 
        - valid loss: 0.3502
    - unexcepted result
        - after all mixup might not be good choice
    
- roll back to exp118 as base
    - full presicion training
    - 0.2 of ratio for warming up
    - GaussNoise

- **exp146**
    - based: exp118
    - full precision training
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - *inititalize lstm head*
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4146, LB: 
    
- **exp147**
    - based: exp118
    - full precision training
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4047, LB: 
    
- **exp148**
    - based: exp118
        - *validation on fold 3*
    - full precision training
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - *removed Noise*
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] fold0 CV: 0.4474, LB: 
    - quite good on fold 3
    
- task for tmr
    - check again with fold 3 with exp147 setup

# 9.19 

- **exp149**
    - based: exp118
    - full precision training
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - MixUp
    - [RESULT] fold0 CV: 0.4325, LB: 0.48
    
- **exp150**
    - based: exp148
    - *4 folds*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *warm up ratio 0.1*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Distortion
        - CroarseDropout (Cutout)
        - MixUp
    - [RESULT] 4 folds CV: 0.4348, LB: 0.46
    
- increase size of image might help
    - right now is 128
    - gradually increase size to see its improvement    

- task for tmr
    - check why exp150 a lot worse than exp114

# 9.20
    
- **exp151**
    - based: exp148
    - *4 folds*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *warm up ratio 0.2*
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - *Noise*
        - Distortion
        - CroarseDropout (Cutout)
        - *No MixUp*
    - [RESULT] 4 folds CV: 0.4373, LB: 0.45

- task for tmr
    - update model code taking for infer notebook
        - done
    - try replicate exp118 on kaggle
    - try out convnextv2_tiny 384?
        - find exp that score around 0.39 and follow it

# 9.21
    
- **exp153**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: , LB:
    - Do not need to finish exp since it it exactly the same as exp118
        - so reproducibility is checked


- Now, these things is a bit concerning
    - is 5 heads necessary?
        - yes (exp154)
    - is batchnorm dropout in cls head necessary?
        - yes 
    - add spartial dropout?
        - done, not improving

- exp154 will be checking necessity of 5 heads
    - by combined every head to 1 head and output shape of (bs, cls)
    
- **exp154**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - *not use 5 head separately for each symptom, but only 1 head*
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4271, LB:
    
- **exp155**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
            - *this time use linear->relu->dropout->linear*
            - *previously was linear->batchnorm->dropout->leakyrelu->liear*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4264, LB:
    
- **exp156**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
        - *add SpatialDropout*
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4127, LB: 0.46

- did not finish exp157

# 9.22
    
- **exp158**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4086, LB:

- taking data for colab A100 credit usage
    
- **exp159**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - *image size 128 to 224*
        - *384 is OOM on A100*
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: , LB:
    - credit
        - at start:           38.41 (@16:37)
        - at train start:     37.12 (@16:47)
        - done train 1 fold:  15.54 (@18:43)
    
- **exp160**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - *epoch 10 to 15*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: 0.4170, LB:

- sticking with exp118 setup might be the best choice for now
- time to look more to ensembling and post-processing

- task for tmr
    - check predicted relative coord
        - done
    - explore option for post-processing

- **exp161**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - *not grouping left, mid, right*
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: , LB:
    - did not finish
    
- **exp162**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - *use max pooling*
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: , LB:
    - did not finish

- **exp163**
    - based: exp127
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] fold0 CV: , LB:

- **exp164**
    - based: exp118
        - *4 folds*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 folds CV: 0.44082, LB: 0.45

# 9.23

- **exp164**
    - based: exp118
        - *4 folds*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from lstm head
        - attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - randombrightnesscontrast
        - blur
        - noise
        - distortion
        - croarsedropout (cutout)
    - [result] 4 folds cv: 0.44082, lb: 0.45

# 9.24

- **exp165**
    - based: exp118
        - *4 folds*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - *mean_max head*
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 folds CV: 0.4364, LB: 

# 9.25

- **exp166**
    - based: exp118
        - *4 folds*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - *avg head*
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 folds CV: 0.4347, LB:


- what's next
    - try different arch
        - convnext
        - maxvit
    - explore option on segmentation head again
        - use disc seg?

# 9.26

- task for tmr
    - work on segmentation head


# 9.27

- disc segmentation can be improve?
    - adding more augmentation might help

- **exp167**
    - based: exp118
    - *use convnextv2_tiny.fcmae_ft_in22k_in1k*
    - epoch 10
    - warm up ratio 0.2
    - *lr 4e-5*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: , LB:
    - fold 3 actting weird, maybe gradient exploding?

# 9.28

- **exp168**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - *change leakyrelu to relu*
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4087, LB:


- **exp169**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.3993, LB:


- **exp170**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4131, LB:

- **exp171**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 2e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4006, LB: 0.45

# 9.29

- **exp172**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - *epoch 15*
    - warm up ratio 0.2
    - *lr 2e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4074, LB:

- **exp173**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 1e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4122, LB:

- **exp174**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 2e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4222, LB:

- **exp175**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 2e-4*
    - fixed left and right mixed up 
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
        - *MixUp*
    - [RESULT] folds0 CV: 0.4207, LB:

- felt like nothing I can do with hparams or augmentation anymore

- task for tmr
    - explore new options 

# 9.30

- **exp176**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - *label_smoothing 0.1*
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4362, LB:

# 10.1

- **exp177**
    - based: exp118
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
        - *add offset move center by 0.3 to top right*
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: , LB:
    - did not finish

- maybe I can find a better way to calculate image of foramina

# 10.2

- **exp178**
    - based: exp118
    - *using datset v7*
        - *included null labels predicted from exp114*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4321, LB: 0.46

- **exp178**
    - based: exp118
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 3e-4*
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4321, LB: 0.46

- **exp179**
    - based: exp118
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 3e-4*
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4349, LB: 

- **exp180**
    - based: exp118
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - warm up ratio 0.2
    - *lr 3e-4*
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4485, LB: 

# 10.3

- **exp181**
    - based: exp118
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *batch size 4*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: , LB: 
    - did not finish

- **exp182**
    - based: exp118
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *batch size 16*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4257, LB: 

- accidentally skipped exp183

- **exp184**
    - based: exp118
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *batch size 16*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4227, LB: 

- **exp185**
    - based: exp118
    - *using datset v6*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *batch size 16*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: , LB: 

# 10.4

- **exp187**
    - based: exp184
        - *4 folds*
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *batch size 16*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 fold CV: 0.4293, LB: 

# 10.5

- **exp188**
    - based: exp184
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *batch size 16*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - *window ratio 0.13*
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4334, LB: 

- **exp189**
    - based: exp184
    - *using datset v7*
    - use tf_efficientnetv2_s.in21k_ft_in1k
    - epoch 10
    - *batch size 16*
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - *window ratio 0.11*
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - *VerticalFlip*
        - *ShiftScaleRotate*
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] folds0 CV: 0.4395, LB: 

- **exp190**
    - based: exp184
    - *using datset v6*
    - use 
    - epoch 10
    - batch size 8
    - warm up ratio 0.2
    - lr 4e-4
    - fixed left and right mixed up
    - window ratio 0.12
    - image size 128
    - removedd null labels
    - use 5 head separately for each symptom
        - each head recieve extracted features from LSTM head
        - Attention head
        - logits is now modified to have dropout
    - group left, mid, right
    - use avg pooling
    - expanding middle (spinal) from 3 to 5
    - 25 slices
        - each 11 from sagittal t1 and t2
        - 3 axial t2
            - use coords as center when crop
            - window ratio size 0.35
    - add augment
        - RandomBrightnessContrast
        - Blur
        - Noise
        - Distortion
        - CroarseDropout (Cutout)
    - [RESULT] 4 folds CV: , LB: 

# 10.9

- compeition ended
    - public rank  62 (silver medal range)
    - private rank 253 (out of mdel range)