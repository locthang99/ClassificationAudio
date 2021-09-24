from pydub import AudioSegment
from os import path
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import math

def convert(file_path, filename):
    SAMPLE_RATE = 22050
    TRACK_DURATION = 60  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    num_segments=10
    num_mfcc=13
    n_fft=2048
    hop_length=512
    # samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    samples_per_segment =int(SAMPLE_RATE * TRACK_DURATION / num_segments)
    num_mfcc_vectors_per_segment = 259
    
    # load audio file
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    data = {
        "mfcc":[]
    }
    # process all segments of audio file
    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(
            signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            print("{}, segment:{}".format(file_path, d+1))


    # data['mfcc'] = MFCCs.tolist()
    with open(filename+".json", "w") as fp:
        json.dump(data, fp, indent=4)


def cut(file_path, file_name):
    dst_wav = file_name + ".wav"
    sound = AudioSegment.from_mp3(file_path)
    sound.duration_seconds
    startTime = 1*0*1000
    endTime = 1*30*1000
    extract = sound[startTime:endTime]
    extract.export(dst_wav, format="wav")
    # Convert
    # convert(dst_wav, file_name)


if __name__ == "__main__":
    #convert("Data/test60.wav","Output/test60mp3")
    #cut("Data/test.mp3","test30")
    #convert("vip4.wav","test1")
    convert("vip7.wav","test7")