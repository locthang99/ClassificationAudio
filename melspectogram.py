import json
import os
import math
from re import T

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """
    # dictionary to store mapping, labels, and MFCCs

    samples_per_segment = SAMPLES_PER_TRACK
    file_path = "MotCoiDiVe.mp3"

    signal, sample_rate = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(signal, sr=sample_rate,n_mfcc=20)
    mfcc = mfcc.T

    print(sample_rate)
    # num_segments = int ((len(signal)-0.5*samples_per_segment)/samples_per_segment)
    # print(num_segments)
    # start = int(samples_per_segment * (num_segments-0.5))
    # finish = start + samples_per_segment

    mel  = librosa.feature.mfcc(signal[SAMPLE_RATE*33:SAMPLE_RATE*43], sr=sample_rate,n_mfcc=20)

    #mel = librosa.feature.melspectrogram(signal[SAMPLE_RATE*0:441], sample_rate)
    mel.T

    print(shape( mel.tolist()))
    log_mel_spectrogram = librosa.power_to_db(mel)
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spectrogram, 
                            x_axis="time",
                            y_axis="mel", 
                            sr=sample_rate)
    plt.colorbar(format="%+2.f")
    plt.show()
    # print(shape(mel.tolist()))

    # process all segments of audio file
    # for d in range(num_segments):
    #     # calculate start and finish sample for current segment
    #     start = samples_per_segment * d
    #     finish = start + samples_per_segment
    #     # extract mfcc
    #     mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    #     mfcc = mfcc.T
    #     # store only mfcc feature with expected number of vectors
    #     if len(mfcc) == num_mfcc_vectors_per_segment:
    #         #print(num_mfcc_vectors_per_segment)
    #         data["mfcc"].append(mfcc.tolist())
    #         print("{}, segment:{}".format(file_path, d+1))

    # print("Dump file ")
    # with open("/content/gdrive/MyDrive/KLTN/Temp/test_mfcc2"+".json", "w+") as fp:
    #     json.dump(data, fp, indent=4)
            
if __name__ == "__main__":
  save_mfcc()