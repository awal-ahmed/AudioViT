# AudioViT

### Prerequisite
* Anaconda

If you need to crop all your audio in same length follow instruction mentioned in [this repository](https://github.com/awal-ahmed/Danger-detection).

### Managing virtual environment
Create a conda environment:
```
conda env create -f audioViT_env.yml
```

Activate the environment:
```
conda activate audioViT_env
```

### To reduce noise from audio
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
&emsp;&emsp;&emsp;Example: --src_root=./DangerDetection<br />
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
&emsp;&emsp;&emsp;Example: --dst_root=CleanData<br />
* --noise_reducer, -nr: Mention the name of noise reduction needed to be used.<br />
&emsp;&emsp;&emsp;&emsp;&emsp;Options: butter, noise_reduce, deNoise, power, centroid_s, centroid_mb, mfcc_up, mfcc_down, median<br />
&emsp;&emsp;&emsp;Example: --noise_reducer=median or -nr=median
* --sr: Mention the sampling rate you want to resize the audio.<br />
&emsp;&emsp;&emsp;Example: --sr=40000


              
