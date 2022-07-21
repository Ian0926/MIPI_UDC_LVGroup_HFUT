# MIPI Challenge 2022 Team LVGroup_HFUT

> This repository is the official [MIPI Challenge 2022](http://mipi-challenge.org/#) implementation of Team LVGroup_HFUT in [Image Restoration for Under-display Camera](https://codalab.lisn.upsaclay.fr/competitions/4874).

## Usage

### Single image inference

`cd your/script/path`

`python infer.py --data_source your/dataset/path --model_path ../pretrained/optimal.pth --save_image --experiment your-experiment`

### Train

`cd your/script/path`

`python train.py --data_source your/dataset/path --experiment your-experiment`

### Test
`cd your/script/path`

`python test.py --data_source your/dataset/path --model_path ../pretrained/optimal.pth --experiment your-experiment`

### Dataset format

The format of the dataset should meet the following code in datasets.py:

`self.img_paths = sorted(glob.glob(data_source + '/' + mode + '/input' + '/*.*'))`

`self.gt_paths = sorted(glob.glob(data_source + '/'  + mode + '/GT' + '/*.*'))`

or

`self.img_paths = sorted(glob.glob(data_source + '/' + 'test' + '/input' + '/*.*'))`

***data_source*** is given by the command line.

***mode*** can be 'train' or 'val'.

### Path to saving results

***when training and validating:***  the default path is `'../results/your-experiment'`

***when testing:***  the default path is `'../outputs/your-experiment/test'`

***when inferring:***  the default path is `'../outputs/your-experiment/infer'`
