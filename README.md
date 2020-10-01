# Efficient-3DCNNs
PyTorch Implementation of the article "[Resource Efficient 3D Convolutional Neural Networks](https://arxiv.org/pdf/1904.02422.pdf)", codes and pretrained models.

## Update!

3D ResNet and 3D ResNeXt models are added! The details of these models can be found in [link](https://arxiv.org/pdf/1711.09577.pdf).

## Requirements

* [PyTorch 1.0.1.post2](http://pytorch.org/)
* OpenCV
* FFmpeg, FFprobe
* Python 3

## Pre-trained models

Pretrained models can be downloaded from [here](https://drive.google.com/open?id=1eggpkmy_zjb62Xra6kQviLa67vzP_FR8).

Implemented models:
 - 3D SqueezeNet
 - 3D MobileNet
 - 3D ShuffleNet
 - 3D MobileNetv2
 - 3D ShuffleNetv2
 
 For state-of-the-art comparison, the following models are also evaluated:
 - ResNet-18
 - ResNet-50
 - ResNet-101
 - ResNext-101
 
 All models (except for SqueezeNet) are evaluated for 4 different complexity levels by adjusting their 'width_multiplier' with 2 different hardware platforms.

## Results

<p align="center"><img src="https://github.com/okankop/Efficient-3DCNNs/blob/master/utils/results.jpg" align="middle" width="900" title="Results of Efficient 3DCNNs" /></p>

## Dataset Preparation

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Different from the other datasets, we did not extract frames from the videos. Insted, we read the frames directly from videos using OpenCV throughout the training. If you want to extract the frames for Kinetics dataset, please follow the preperation steps in [Kensho Hara's codebase](https://github.com/kenshohara/3D-ResNets-PyTorch). You also need to modify the kinetics.py file in the datasets folder.

* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python utils/kinetics_json.py train_csv_path val_csv_path video_dataset_path dst_json_path
```

### Jester

* Download videos [here](https://20bn.com/datasets/jester#download).
* Generate n_frames files using ```utils/n_frames_jester.py```

```bash
python utils/n_frames_jester.py dataset_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/jester_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist.txt, vallist.txt

```bash
python utils/jester_json.py annotation_dir_path
```

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```


## Running the code

Model configurations are given as follows:

```misc
ShuffleNetV1-1.0x : --model shufflenet   --width_mult 1.0 --groups 3
ShuffleNetV2-1.0x : --model shufflenetv2 --width_mult 1.0
MobileNetV1-1.0x  : --model mobilenet    --width_mult 1.0
MobileNetV2-1.0x  : --model mobilenetv2  --width_mult 1.0 
SqueezeNet	  : --model squeezenet --version 1.1
ResNet-18	  : --model resnet  --model_depth 18  --resnet_shortcut A
ResNet-50	  : --model resnet  --model_depth 50  --resnet_shortcut B
ResNet-101	  : --model resnet  --model_depth 101 --resnet_shortcut B
ResNeXt-101	  : --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
```

Please check all the 'Resource efficient 3D CNN models' in models folder and run the code by providing the necessary parameters. An example run is given as follows:

- Training from scratch:
```bash
python main.py --root_path ~/ \
	--video_path ~/datasets/jester \
	--annotation_path Efficient-3DCNNs/annotation_Jester/jester.json \
	--result_path Efficient-3DCNNs/results \
	--dataset jester \
	--n_classes 27 \
	--model mobilenet \
	--width_mult 0.5 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--downsample 2 \
	--batch_size 64 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
```

- Resuming training from a checkpoint:
```bash
python main.py --root_path ~/ \
	--video_path ~/datasets/jester \
	--annotation_path Efficient-3DCNNs/annotation_Jester/jester.json \
	--result_path Efficient-3DCNNs/results \
	--resume_path Efficient-3DCNNs/results/jester_shufflenet_0.5x_G3_RGB_16_best.pth \
	--dataset jester \
	--n_classes 27 \
	--model shufflenet \
	--groups 3 \
	--width_mult 0.5 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--downsample 2 \
	--batch_size 64 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
```


- Training from a pretrained model. Use '--ft_portion' and select 'complete' or 'last_layer' for the fine tuning:
```bash
python main.py --root_path ~/ \
	--video_path ~/datasets/jester \
	--annotation_path Efficient-3DCNNs/annotation_UCF101/ucf101_01.json \
	--result_path Efficient-3DCNNs/results \
	--pretrain_path Efficient-3DCNNs/results/kinetics_shufflenet_0.5x_G3_RGB_16_best.pth \
	--dataset ucf101 \
	--n_classes 600 \
	--n_finetune_classes 101 \
	--ft_portion last_layer \
	--model shufflenet \
	--groups 3 \
	--width_mult 0.5 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--downsample 1 \
	--batch_size 64 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
```

### Augmentations

There are several augmentation techniques available. Please check spatial_transforms.py and temporal_transforms.py for the details of the augmentation methods.

Note: Do not use "RandomHorizontalFlip" for trainings of Jester dataset, as it alters the class type of some classes (e.g. Swipe_Left --> RandomHorizontalFlip() --> Swipe_Right)

### Calculating Video Accuracy

In order to calculate viceo accuracy, you should first run the models with '--test' mode in order to create 'val.json'. Then, you need to run 'video_accuracy.py' in utils folder to calculate video accuracies. 

### Calculating FLOPs

In order to calculate FLOPs, run the file 'calculate_FLOP.py'. You need to fist uncomment the desired model in the file. 

## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@inproceedings{kopuklu2019resource,
  title={Resource efficient 3d convolutional neural networks},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Kose, Neslihan and Gunduz, Ahmet and Rigoll, Gerhard},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={1910--1919},
  year={2019},
  organization={IEEE}
}
```

## Acknowledgement
We thank Kensho Hara for releasing his [codebase](https://github.com/kenshohara/3D-ResNets-PyTorch), which we build our work on top.
