# [CVPR'2023] - SeqTrack: Sequence to Sequence Learning for Visual Object Tracking

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqtrack-sequence-to-sequence-learning-for/visual-object-tracking-on-tnl2k)](https://paperswithcode.com/sota/visual-object-tracking-on-tnl2k?p=seqtrack-sequence-to-sequence-learning-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqtrack-sequence-to-sequence-learning-for/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=seqtrack-sequence-to-sequence-learning-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqtrack-sequence-to-sequence-learning-for/visual-object-tracking-on-lasot-ext)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot-ext?p=seqtrack-sequence-to-sequence-learning-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqtrack-sequence-to-sequence-learning-for/visual-object-tracking-on-trackingnet)](https://paperswithcode.com/sota/visual-object-tracking-on-trackingnet?p=seqtrack-sequence-to-sequence-learning-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqtrack-sequence-to-sequence-learning-for/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=seqtrack-sequence-to-sequence-learning-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqtrack-sequence-to-sequence-learning-for/visual-object-tracking-on-uav123)](https://paperswithcode.com/sota/visual-object-tracking-on-uav123?p=seqtrack-sequence-to-sequence-learning-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seqtrack-sequence-to-sequence-learning-for/visual-object-tracking-on-needforspeed)](https://paperswithcode.com/sota/visual-object-tracking-on-needforspeed?p=seqtrack-sequence-to-sequence-learning-for)

> [**SeqTrack: Sequence to Sequence Learning for Visual Object Tracking**](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SeqTrack_Sequence_to_Sequence_Learning_for_Visual_Object_Tracking_CVPR_2023_paper.html)<br>
> accepted by CVPR2023<br>
> [Xin Chen](https://scholar.google.com.hk/citations?user=A04HWTIAAAAJ&hl=zh-CN&oi=sr), [Houwen Peng](https://houwenpeng.com/), [Dong Wang](http://faculty.dlut.edu.cn/wangdongice/zh_CN/index.htm), [Huchuan Lu](https://ice.dlut.edu.cn/lu/), [Han Hu](https://ancientmooner.github.io/)


This is an official pytorch implementation of the CVPR2023 paper SeqTrack: Sequence to Sequence Learning for Visual Object Tracking, a new framework for visual object tracking.




## Highlights
### Seq2seq modeling
SeqTrack models tracking as a **sequence generation** task. If the model knows where the target object is, we could simply teach it how to read the bounding box out.

![SeqTrack_pipeline](tracking/pipeline.gif)

### Simple architecture and loss function
SeqTrack only adopts a **plain encoder-decoder transformer** architecture with a **simple cross-entropy loss**.

![SeqTrack_Framework](tracking/Framework.png)

### Strong performance
| Tracker      | LaSOT (AUC) | GOT-10K (AO) | TrackingNet (AUC) |
|--------------|-------------|--------------|-------------------|
| **SeqTrack** | **72.5**    | **74.8**     | **85.5**          |
| OSTrack      | 71.1        | 73.7         | 83.9              |
| SimTrack     | 70.5        | 69.8         | 83.4              |
| Mixformer    | 70.1        | 70.7         | 83.9              |

## Install the environment
```
conda create -n seqtrack python=3.8
conda activate seqtrack
bash install.sh
```

* Add the project path to environment variables
```
export PYTHONPATH=<absolute_path_of_SeqTrack>:$PYTHONPATH
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${SeqTrack_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train SeqTrack
```
python -m torch.distributed.launch --nproc_per_node 8 lib/train/run_training.py --script seqtrack --config seqtrack_b256 --save_dir .
```

(Optionally) Debugging training with a single GPU
```
python tracking/train.py --script seqtrack --config seqtrack_b256 --save_dir . --mode single
```


## Test and evaluate on benchmarks

- LaSOT
```
python tracking/test.py seqtrack seqtrack_b256 --dataset lasot --threads 2
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py seqtrack seqtrack_b256_got --dataset got10k_test --threads 2
python lib/test/utils/transform_got10k.py --tracker_name seqtrack --cfg_name seqtrack_b256_got
```
- TrackingNet
```
python tracking/test.py seqtrack seqtrack_b256 --dataset trackingnet --threads 2
python lib/test/utils/transform_trackingnet.py --tracker_name seqtrack --cfg_name seqtrack_b256
```
- TNL2K
```
python tracking/test.py seqtrack seqtrack_b256 --dataset trackingnet --threads 2
python tracking/analysis_results.py # need to modify tracker configs and names
```
- UAV123
```
python tracking/test.py seqtrack seqtrack_b256 --dataset uav --threads 2
python tracking/analysis_results.py # need to modify tracker configs and names
```
- NFS
```
python tracking/test.py seqtrack seqtrack_b256 --dataset nfs --threads 2
python tracking/analysis_results.py # need to modify tracker configs and names
```
- VOT2020  
Before evaluating "SeqTrack+AR" on VOT2020, please install some extra packages following [external/AR/README.md](external/AR/README.md)
```
cd external/vot20/<workspace_dir>
export PYTHONPATH=<path to the seqtrack project>:$PYTHONPATH
vot evaluate --workspace . seqtrack_b256_ar
vot analysis --nocache
```


## Test FLOPs, Params, and Speed
```
# Profiling SeqTrack-B256 model
python tracking/profile_model.py --script seqtrack --config seqtrack_b256
```

## Model Zoo
The trained models, and the raw tracking results are provided in the [model zoo](MODEL_ZOO.md)

## Acknowledgement
* This codebase is implemented on [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking) libraries, also refers to [Stable-Pix2Seq](https://github.com/gaopengcuhk/Stable-Pix2Seq), and borrows [AlphaRefine](https://github.com/MasterBin-IIAU/AlphaRefine) for VOT evaluation. 
We would like to thank their authors for providing great libraries.




# Carotid Artery Implenentation

## Data Preparation Process

To ensure the proper functionality of the code, it was necessary to create a structure suitable for the default data. As a result, structures appropriate for the data loaders were added, which can be found in the SeqTrack/data/.. directory. Except for the carotidartery folder, the other folders were added to enable the project to run correctly and do not have any additional effects.
Modifications in YAML Files

### In the .yaml files (located in SeqTrack/experiments/seqtrack/..):

- Enabled the Carotidartery dataset and disabled other datasets.
- Set the epoch numbers appropriately.
- Adjusted the batch sizes accordingly.
- Changes were made to the seqtrack_b256.yaml file as seqtrack_b256 was being used.

Since the project was initially developed on Windows and later on Linux, Windows paths were commented out rather than deleted to preserve the option to run the project on Windows in the future. These changes are indicated with comments "#WINDOWS" and "#LINUX".


Path Adjustments

The files SeqTrack/lib/test/evaluation/local.py and SeqTrack/lib/train/admin/local.py are populated using the command from the SeqTrack project's README under the "Set project paths" section, with manual adjustments made for the carotidartery dataset as needed:
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

Environment Adjustments

    SeqTrack/lib/train/admin/environment.py was updated to include the carotidartery dataset in the environment.

Data Loader Functions

    SeqTrack/lib/train/base_functions.py was updated with functions for the carotidartery data loader.
    SeqTrack/lib/train/data_specs/carotidartery_train_split.txt contains the names of training sequences.
    SeqTrack/lib/train/dataset/carotidartery.py serves as the data loader for the carotidartery dataset, modeled after other data loaders like the "lasot" data loader.


Data Preparation and Manipulation

Scripts for data manipulation are located in the SeqTrack/data_preparation_scripts/.. directory:

    image_reader.py: Helps read label images into a data structure.
    nifti_reader.py: Assists in reading .nii files of nifti data type (currently not in use but not deleted).
    png_to_jpg.py: Converts images to jpg format as loaders accept only jpg files.
    rename_images.py: Renames images in each sequence starting from 00000001 in sequential order.
    seg_to_bb.py: Converts segmentation format labels to bounding box format and writes to a txt file (groundtruth files are named "label_vessel.txt").

Training Process

The training process generates models saved as checkpoints after specific cycles. Training can continue from these checkpoints or can be used for testing.

    SeqTrack/lib/train/trainers/base_trainer.py was modified to handle the loading of pretrained models. Adjustments were made to ensure parameters missing in the default pretrained model do not cause issues when continuing training.
    SeqTrack/lib/train/trainers/ltr_trainer.py was modified to plot loss and IoU values, saved in the Seqtrack/charts/.. directory.

Testing Process

    SeqTrack/lib/test/evaluation/carotidarterydataset.py contains the main test class for the carotidartery dataset. A significant step here is the reduction of groundtruth bounding boxes by 10% to their original size using the shrink_bounding_box function. The list of test sequences is in the _get_sequence_list function.
    SeqTrack/lib/test/evaluation/environment.py was updated to include the carotid artery test environment.
    SeqTrack/lib/test/evaluation/tracker.py saw significant changes for data visualization. The modifications allow for adding predictions and groundtruths to image frames and subsequently creating videos. Image and video outputs are saved in the Seqtrack/test/.. directory under visual_results and video_results folders.

Analysis Process

    SeqTrack/tracking/analysis_results.py was updated to generate analysis results for the carotid artery dataset.