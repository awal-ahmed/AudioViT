import librosa
import argparse
import os
import wavio
import pywt
import soundfile
import math
import python_speech_features
import numpy as np
import scipy as sp
import noisereduce as nr
from pysndfx import AudioEffectsChain
from scipy import signal
from scipy.io import wavfile
from glob import glob
from librosa.core import resample, to_mono
from tqdm import tqdm

# http://python-speech-features.readthedocs.io/en/latest/
# https://github.com/jameslyons/python_speech_features
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#deltas-and-delta-deltas


# http://dsp.stackexchange.com/search?q=noise+reduction/


'''------------------------------------
NOISE REDUCTION USING POWER:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_power(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.1

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
    y_clean = less_noise(y)

    return y_clean


'''------------------------------------
NOISE REDUCTION USING CENTROID ANALYSIS:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''

def reduce_noise_centroid_s(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = 20

    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5).limiter(gain=6.0)

    y_cleaned = less_noise(y)

    return y_cleaned* 10000


def reduce_noise_centroid_mb(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = 20

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5).highshelf(gain=-30.0, frequency=threshold_h, slope=0.5).limiter(gain=10.0)
    y_cleaned = less_noise(y)

    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows/3*2)

    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted


'''------------------------------------
NOISE REDUCTION USING MFCC:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_mfcc_down(y, sr):

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    min_hz = min(hz)

    speech_booster = AudioEffectsChain().highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.6).limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)


def reduce_noise_mfcc_up(y, sr):

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted*10000)


'''------------------------------------
NOISE REDUCTION USING MEDIAN:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''

def reduce_noise_median(y, sr):
    y = sp.signal.medfilt(y,3)
    return (y)


'''------------------------------------
SILENCE TRIMMER:
    receives an audio matrix,
    returns an audio matrix with less silence and the amout of time that was trimmed
------------------------------------'''
def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y=y) - librosa.get_duration(y=y_trimmed)

    return y_trimmed, trimmed_length


'''------------------------------------
AUDIO ENHANCER:
    receives an audio matrix,
    returns the same matrix after audio manipulation
------------------------------------'''
def enhance(y):
    apply_audio_effects = AudioEffectsChain().lowshelf(gain=10.0, frequency=260, slope=0.1).reverb(reverberance=25, hf_damping=5, room_scale=5, stereo_depth=50, pre_delay=20, wet_gain=0, wet_only=False)#.normalize()
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


'''------------------------------------
OUTPUT GENERATOR:
    receives a destination path, file name, audio matrix, and sample rate,
    generates a wav file based on input
------------------------------------'''
def output_file(destination ,filename, y, sr, ext=""):
    destination = destination + filename[:-4] + ext + '.wav'
    librosa.output.write_wav(destination, y, sr)


def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, fs, cutoff = 10, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def save_sample(sample, rate, target_dir, fn):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'.wav')
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variability of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


class AudioDeNoise:
    """
    Class to de-noise the audio signal. The audio file is read in chunks and processed,
    cleaned and appended to the output file..
    It can de-noise multiple channels, any sized file, formats supported by soundfile
    Wavelets used ::
        Daubechies 4 : db4
        Level : decided by pyWavelets
    Attributes
    ----------
    __inputFile : str
        name of the input audio file
    Examples
    --------
    To de noise an audio file
    >>> audioDenoiser = AudioDeNoise("input.wav")
    >>> audioDenoiser.deNoise("input_denoised.wav")
    To generate the noise profile
    >>> audioDenoiser = AudioDeNoise("input.wav")
    >>> audioDenoiser.generateNoiseProfile("input_noise_profile.wav")
    """

    def __init__(self, inputFile):
        self.__inputFile = inputFile
        self.__noiseProfile = None

    def deNoise(self):
        """
        De-noising function that reads the audio signal in chunks and processes
        and writes to the output file efficiently.
        VISU Shrink is used to generate the noise threshold
        Parameters
        ----------
        outputFile : str
            de-noised file name
        """
        info = soundfile.info(self.__inputFile)  # getting info of the audio
        rate = info.samplerate

        
        for block in tqdm(soundfile.blocks(self.__inputFile, int(rate * info.duration * 0.10))):
            coefficients = pywt.wavedec(block, 'db4', mode='per', level=2)

            #  getting variance of the input signal
            sigma = mad(coefficients[- 1])

            # VISU Shrink thresholding by applying the universal threshold proposed by Donoho and Johnstone
            thresh = sigma * np.sqrt(2 * np.log(len(block)))

            # thresholding using the noise threshold generated
            coefficients[1:] = (pywt.threshold(i, value=thresh, mode='soft') for i in coefficients[1:])

            # getting the clean signal as in original form and writing to the file
            clean = pywt.waverec(coefficients, 'db4', mode='per')
            
            clean = clean*10000
            return clean
    

def noise_reduction(args):
    src_root = args.src_root
    dst_root = args.dst_root
    noise_reducer = args.noise_reducer
    dirs = os.listdir(src_root)
    check_dir(dst_root)

    print('Name of noise_reducer: ', noise_reducer)

    for dir in dirs:
        check_dir(os.path.join(dst_root, dir))
        classes = os.listdir(os.path.join(src_root, dir))

        for _cls in classes:
            target_dir = os.path.join(dst_root, dir, _cls)
            check_dir(target_dir)

            src_dir = os.path.join(src_root, dir, _cls)
            for fn in tqdm(os.listdir(src_dir)):
                path = os.path.join(src_dir, fn)
                
                obj = wavio.read(path)
                y = obj.data.astype(np.float32, order='F')
                sr = obj.rate

                try:
                    channel = y.shape[1]
                    if channel == 2:
                        y = to_mono(y.T)
                    elif channel == 1:
                        y = to_mono(y.reshape(-1))
                except IndexError:
                    y = to_mono(y.reshape(-1))
                    pass
                except Exception as exc:
                    raise exc
                
                if noise_reducer == 'butter':
                    wav = butter_highpass_filter(y, sr)
                elif noise_reducer == 'noise_reduce':
                    wav =  nr.reduce_noise(y=y, sr=sr)
                elif noise_reducer == 'deNoise':
                    audioDenoiser = AudioDeNoise(inputFile=path)
                    wav = audioDenoiser.deNoise()
                elif noise_reducer == 'power':
                    wav = reduce_noise_power(y, sr)
                    wav, time_trimmed = trim_silence(wav)
                elif noise_reducer == 'centroid_s':
                    wav = reduce_noise_centroid_s(y, sr)
                    wav, time_trimmed = trim_silence(wav)
                elif noise_reducer == 'centroid_mb':
                    wav = reduce_noise_centroid_mb(y, sr)
                    wav, time_trimmed = trim_silence(wav)
                elif noise_reducer == 'mfcc_up':
                    wav = reduce_noise_mfcc_up(y, sr)
                    wav, time_trimmed = trim_silence(wav)
                elif noise_reducer == 'mfcc_down':
                    wav = reduce_noise_mfcc_down(y, sr)
                    wav, time_trimmed = trim_silence(wav)
                elif noise_reducer == 'median':
                    wav = reduce_noise_median(y, sr)
                    wav, time_trimmed = trim_silence(wav)
                else:
                    raise Exception('Please choose a correct noise reduction method')
                
                wav = resample(y=wav, orig_sr=wav.shape[0], target_sr=sr)
                wav= np.reshape(wav,(wav.shape[0],1))
                save_sample(wav, args.sr, target_dir, fn)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='./DangerDetection',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='CleanData',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--noise_reducer','-nr', type=str, default='median',
                        help='Noise reduction to choose: butter, noise_reduce, deNoise, power, centroid_s, \
                            centroid_mb, mfcc_up, mfcc_down, median')
    parser.add_argument('--sr', type=int, default=40000,
                        help='rate to downsample audio')
    args, _ = parser.parse_known_args()
    noise_reduction(args)
