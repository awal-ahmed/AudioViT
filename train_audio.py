import os
import argparse
import warnings
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

from glob import glob
from sklearn.model_selection import train_test_split
from models import Conv1D, resnet50, LSTM, mobilenetv2, inceptionv3, xception, resnet101, dencenet, Conv2D, vgg19
from vit import AudioViT, ClassToken


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.set_visible_devices(physical_devices[0:1], 'GPU')
warnings.filterwarnings("ignore")

glo_acc=0.0

""" Hyperparameters """
hp = {}
hp["image_size"] = 250
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 4
hp["lr"] = 1e-4
hp["num_epochs"] = 3
hp["class_names"] = ["Child", "Normal", "Women"]

hp["num_layers"] = 6
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    
def train(args):
    
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_name = args.model_type
    noise_reducer = args.noise_reduce
    hp["batch_size"] = batch_size
    
    params = {'N_CLASSES':len(os.listdir(src_root)),
              'SR':sr,
              'DT':dt}
    models = {'resnet50':resnet50(**params),
              'resnet101':resnet101(**params),
              'lstm':LSTM(**params),
              'mobilenetv2':mobilenetv2(**params),
              'inceptionv3':inceptionv3(**params),
              'xception':xception(**params),
              'dencenet':dencenet(**params),
              'conv2d': Conv2D(**params), 
              'conv1d': Conv1D(**params),
              'vgg19': vgg19(**params),
              'audiovit':AudioViT(hp, params)}
    
    csv_path = os.path.join('./testoutput', '{}_history.csv'.format(model_name+noise_reducer))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.1,
                                                                  random_state=0)


    assert len(label_train) >= batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(len(set(label_val)), params['N_CLASSES']))

    tg = DataGenerator(wav_train, label_train, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)

    model =  models[model_name]
    
    try:
        old_path = args.old + '/' +model_name+'/best_'+model_name+noise_reducer+'.h5'
        model =  load_model(old_path, 
                        custom_objects={'STFT':STFT,
                                        'Magnitude':Magnitude,
                                        'ApplyFilterbank':ApplyFilterbank,
                                        'MagnitudeToDecibel':MagnitudeToDecibel,
                                        'ClassToken': ClassToken})
        print('old model loaded')
    except:
        print('no old model')
    
    class myCallback(tf.keras.callbacks.Callback): 
        def on_epoch_end(self, epoch, logs={}): 
            global glo_acc

            if(logs.get('val_accuracy') > glo_acc ):
                model.save('./testoutput/'+model_name+'/best_'+model_name+noise_reducer+'.h5')
                
    cp = myCallback()

    print('Noise Reducer: ', noise_reducer)
    print('Model Name: ', model_name)
    
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(tg, validation_data=vg,
              epochs=150, verbose=1,
              callbacks=[csv_logger, cp])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='lstm',
                        help='model to run. i.e.  conv1d, mobilenetv2, inceptionv3, xception, \
                            dencenet, resnet50, resnet101, lstm, audiovit, conv2d, vgg19')
    parser.add_argument('--src_root', type=str, default='./CleanData/train',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=40000,
                        help='sample rate of clean audio')
    parser.add_argument('--noise_reduce', type=str, default='median',
                        help='Noise reduction to choose: butter, noise_reduce, deNoise, power, centroid_s, \
                            centroid_mb, mfcc_up, mfcc_down, median')
    parser.add_argument('--old', type=str, default='./testoutput',
                        help='model to run. i.e. deNoise, centroid')
    args, _ = parser.parse_known_args()

    train(args)

print(glo_acc)

