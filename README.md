# AudioViT


### Prerequisite
* Anaconda

## Code with default settings 
Download the audio from [here](https://www.kaggle.com/datasets/awalahmedfime/danger40000).<br /> 
Extract the dataset and move the _DangerDetection_ inside the _AudioViT_ folder.<br />
### To run all files with default settings follow these instructions<br />
```
conda env create -f audioViT_env.yml
conda activate audioViT_env
python noise_cancel.py 
python train_audio.py
python test_audio.py
```

## Code with custome settings
Prepare your own dataset. Move it under _AudioViT_ folder.
If you need to crop all your audio in the same length follow  the instructions mentioned in [this repository](https://github.com/awal-ahmed/Danger-detection).

### Managing virtual environment
Create a conda environment:
```
conda env create -f audioViT_env.yml
```

Activate the environment:
```
conda activate audioViT_env
```

### Reduce noise
To run with the default value
```
python noise_cancel.py 
```
To change the noise reduction models:
```
python noise_cancel.py --noise_reducer=mfcc_up
or
python noise_cancel.py -nr=mfcc_up
```

Options to customize noise reduction
* --src_root: Path till the root of the dataset with noise. <br />
&emsp;&emsp;&emsp;&emsp;&emsp;If the folder structure of the dataset is like this: <br />
&emsp;&emsp;&emsp;&emsp;&emsp;|---DangerDetection<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|---test<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Child<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Normal<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Women<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|---train<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Child<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Normal<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Women<br />
&emsp;&emsp;&emsp;Example: `--src_root=./DangerDetection`<br />
* --dst_root: Path till the root of the dataset where it will be saved. <br />
&emsp;&emsp;&emsp;&emsp;&emsp;It will create folder structure like this: <br />
&emsp;&emsp;&emsp;&emsp;&emsp;|---CleanData<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|---test<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Child<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Normal<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Women<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|---train<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Child<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Normal<br />
&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|---Women<br />
&emsp;&emsp;&emsp;Example: `--dst_root=CleanData`<br />
* --noise_reducer, -nr: Mention the name of noise reduction needed to be used.<br />
&emsp;&emsp;&emsp;&emsp;&emsp;Options: butter, noise_reduce, deNoise, power, centroid_s, centroid_mb, mfcc_up, mfcc_down, median<br />
&emsp;&emsp;&emsp;Example: `--noise_reducer=median` or `-nr=median`
* --sr: Mention the sampling rate you want to resize the audio.<br />
&emsp;&emsp;&emsp;Example: `--sr=40000`

### Train audio signal
To train with the default parameters
```
python train_audio.py
```
To change the noise reduction models:
```
python train_audio.py --model_type=audiovit
```

Options to customize audio training
* --model_type: Mention the model name you wanna use for training.<br />
&emsp;&emsp;&emsp;&emsp;&emsp;Options: conv1d, mobilenetv2, inceptionv3, xception, dencenet, resnet50, resnet101, lstm, audiovit, conv2d, vgg19<br />
&emsp;&emsp;&emsp;Example: `--model_type=audiovit`
* --training_root: Mention the root of the training folder.<br />
&emsp;&emsp;&emsp;Example: `--training_root=./CleanData/train`
* --batch_size: Mention the batch size for trining.<br />
&emsp;&emsp;&emsp;Example: `--batch_size=4`
* --delta_time: Mention the length in seconds of each audio for training.<br />
&emsp;&emsp;&emsp;Example: `--delta_time=1.0`
* --sr: Mention the sampling rate you have resampled your in [noise reduction](https://github.com/awal-ahmed/AudioViT/edit/main/README.md#Reduce-noise).<br />
&emsp;&emsp;&emsp;Example: `--sr=40000`
* --noise_reduce: Mention the noise reducer name you have resampled your in [noise reduction](https://github.com/awal-ahmed/AudioViT/edit/main/README.md#Reduce-noise).<br />
&emsp;&emsp;&emsp;&emsp;&emsp;Options: butter, noise_reduce, deNoise, power, centroid_s, centroid_mb, mfcc_up, mfcc_down, median<br />
&emsp;&emsp;&emsp;Example: `--noise_reduce=median`
* --old: Mention the folder name where you want to save your models.<br />
&emsp;&emsp;&emsp;Example: `--old=./testoutput`

### Test audio signal
To test with the default parameters
```
python test_audio.py
```
To change the noise reduction models:
```
python test_audio.py --model_type=audiovit
```

Options to customize audio testing
* --model_type: Mention the model name you wanna test your data with.<br />
&emsp;&emsp;&emsp;&emsp;&emsp;Options: conv1d, mobilenetv2, inceptionv3, xception, dencenet, resnet50, resnet101, lstm, audiovit, conv2d, vgg19<br />
&emsp;&emsp;&emsp;Example: `--model_type=audiovit`
* --test_dir: Mention the root of the testing folder.<br />
&emsp;&emsp;&emsp;Example: `--test_dir=./CleanData/test`
* --noise_reduce: Mention the noise reducer name you have resampled your in [noise reduction](https://github.com/awal-ahmed/AudioViT/edit/main/README.md#Reduce-noise).<br />
&emsp;&emsp;&emsp;&emsp;&emsp;Options: butter, noise_reduce, deNoise, power, centroid_s, centroid_mb, mfcc_up, mfcc_down, median<br />
&emsp;&emsp;&emsp;Example: `--noise_reduce=median`
* --old: Mention the folder name where you want to save your models.<br />
&emsp;&emsp;&emsp;Example: `--old=./testoutput`



[This paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423021486) might help understand this repository.

If you find this repository and paper helpful, we would appreciate using the following citations:

```
@article{fime2024audio,
  title={Audio signal based danger detection using signal processing and deep learning},
  author={Fime, Awal Ahmed and Ashikuzzaman, Md and Aziz, Abdul},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121646},
  year={2024},
  publisher={Elsevier}
}
```
