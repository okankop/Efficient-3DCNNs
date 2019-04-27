# Efficient-3DCNNs
PyTorch Implementation of the article "[Resource Efficient 3D Convolutional Neural Networks]()", codes and pretrained models.

## Requirements

* [PyTorch 1.0.1.post2](http://pytorch.org/)
* OpenCV
* FFmpeg, FFprobe
* Python 3

## Pre-trained models

Pretrained models can be downloaded from [here](https://drive.google.com/open?id=1eggpkmy_zjb62Xra6kQviLa67vzP_FR8).

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

Please check all the 'Resource efficient 3D CNN models' in models folder and run the code by providing the necessary parameters. An example run is given as follows:

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

### Calculating Video Accuracy

In order to calculate viceo accuracy, you should first run the models with '--test' mode in order to create 'val.json'. Then, you need to run 'video_accuracy.py' in utils folder to calculate video accuracies. 

### Calculating FLOPs

In order to calculate FLOPs, run the file 'calculate_FLOP.py'. You need to fist uncomment the desired model in the file. 

## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@article{kopuklu2019resource,
  title={Resource Efficient 3D Convolutional Neural Networks},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Kose, Neslihan and Gunduz, Ahmet and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:1904.02422},
  year={2019}
}
```

## Acknowledgement
We thank Kensho Hara for releasing his [codebase](https://github.com/kenshohara/3D-ResNets-PyTorch), which we build our work on top.
