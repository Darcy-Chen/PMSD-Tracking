# PMSD-Tracking

## Description
This repository contains code used in the project: Carotid Artery Tracking for Ultrasound Images at Technical University Munich. The code is implemented on top of [SeqTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack).

The code is tested on Ubuntu 20.04 with RTX 3080 using cuda 11.7 and python 3.8.

## Notes

1. Pass through the data to be trained or tested on using keyword **custom**, the dataset should follow the structure:
   ```
   ${SeqTrack_ROOT}
    -- data
        -- custom
            -- train
                -- list.txt
                -- img
                    |-- 0000.txt
                    |-- 0001.png
                    ...
                -- label
                    |-- label.txt
            -- val
                -- list.txt
                -- img
                    |-- 0000.txt
                    |-- 0001.png
                    ...
                -- label
                    |-- label.txt
            -- test
                   -- img
                    |-- groundtruth.txt
                    |-- 0000.png

   ```
2. In list.txt, write down all the folders with images you want to use for training/testing, one folder per line
3. In label.txt, write down all the bounding boxes for the corresponding images [left,top,width,height]
4. In the test folders, provide the ground truth for the first image
5. If your data has class labels, go to lib/train/data/sampler.py uncomment line 170 to 178 and comment out line 179 to 185
6. If you images are in jpg, use jpeg4py as the image reader, otherwise use opencv
7. Modify the parameters in experiment/model.yaml file to change the settings for training/inference
