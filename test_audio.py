import numpy as np
import pandas as pd 
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.set_visible_devices(physical_devices[0:1], 'GPU')
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv1D, resnet50, LSTM, mobilenetv2, inceptionv3, xception, resnet101, dencenet
from vit import AudioViT, ClassToken
from tqdm import tqdm
from glob import glob
import argparse
import warnings
from sklearn import metrics
from scipy import stats
from librosa.core import resample, to_mono
import wavio

from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

import warnings
warnings.filterwarnings("ignore")


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    
    
    #print(wav.shape)
    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc
    #wav = resample(wav, rate, sr)
    #wav = wav.astype(np.int16)
    return sr, wav


def make_prediction(args):
    #model = load_model("../input/danger-detection-2-0/models/mobilenetv20.h5")
    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel,
                        'ClassToken': ClassToken})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    results = []
    co=0
    ch=0
    wo=0
    no=0
    tch=0
    two=0
    tno=0
    child_pred_list = []
    women_pred_list = []
    normal_pred_list = []
    act_child_pred_list = []
    act_women_pred_list = []
    act_normal_pred_list = []
    true_class = []
    pred_class = []
    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):

        rate, wav = wavfile.read(wav_fn)
        wav = np.resize(wav, 40000)
        wav= np.reshape(wav,(1, wav.shape[0],1))
        y_pred = model.predict(wav)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)

        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        if(real_class==classes[y_pred]):
          co=co+1;
        if(real_class=="Child"):
            true_class.append(0)
            tch+=1;
            act_child_pred_list.append(1)
            act_women_pred_list.append(0)
            act_normal_pred_list.append(0)

            if(real_class==classes[y_pred]):
                ch+=1
        elif(real_class=="Normal"):
            true_class.append(2)
            tno+=1;
            act_child_pred_list.append(0)
            act_women_pred_list.append(0)
            act_normal_pred_list.append(1)
            if(real_class==classes[y_pred]):
                no+=1
        if(real_class=="Women"):
            true_class.append(1)
            two+=1;
            act_child_pred_list.append(0)
            act_women_pred_list.append(1)
            act_normal_pred_list.append(0)
            if(real_class==classes[y_pred]):
                wo+=1
        
        if classes[y_pred] == "Child":
            pred_class.append(0)
            child_pred_list.append(1)
            women_pred_list.append(0)
            normal_pred_list.append(0)
        elif classes[y_pred] == "Normal":
            pred_class.append(2)
            child_pred_list.append(0)
            women_pred_list.append(0)
            normal_pred_list.append(1)
        elif classes[y_pred] == "Women":
            pred_class.append(1)
            child_pred_list.append(0)
            women_pred_list.append(1)
            normal_pred_list.append(0)

    results.append(y_mean)

    file1 = open("accuracy.txt", "w")  # append mode

    file1.write("{} {} \n".format(model_name, noise_reducer))
    file1.write('total_accuracy: {} \n'.format(co/len(wav_paths)))
    file1.write('child_accuracy: {} \n'.format(ch/tch))
    file1.write('women_accuracy: {} \n'.format(wo/two))
    file1.write('nornal_accuracy: {}  \n \n'.format(no/tno))


    file1.write("{} {} \n".format(model_name, noise_reducer))
    file1.write('Precision : {} \n'.format(metrics.precision_score(true_class,pred_class, average='macro')))
    file1.write('Recall : {} \n'.format(metrics.recall_score(true_class,pred_class, average='macro')))
    file1.write('F-score : {} \n'.format(metrics.f1_score(true_class,pred_class, average='macro')))
    file1.write('Mean Squared Error(MSE): {} \n'.format(metrics.mean_squared_error(true_class, pred_class)))
    file1.write('t-test : {} \n \n'.format(stats.ttest_ind(a=true_class, b=pred_class, equal_var=True)))
    file1.close()
    file1.close()
    print('total_accuracy: {}'.format(co/len(wav_paths)))
    print('child_accuracy: {}'.format(ch/tch))
    print('women_accuracy: {}'.format(wo/two))
    print('normal_accuracy: {}'.format(no/tno))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='./testoutput/'+model_name+'/best_'+model_name+noise_reducer+'.h5',
                        help='model file to make predictions, eg. resnet50, resnet101, dencenet, mobilenetv2, inceptionv3, xception')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='./CleanData/test',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=40000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_prediction(args)

