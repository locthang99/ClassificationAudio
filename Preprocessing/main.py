import json
import os
import math
from re import T
import librosa
import threading

obj_genre = {
    "viet-nam-cai-luong-":0,
    "viet-nam-nhac-tre-v-pop-":1,
    "viet-nam-nhac-trinh-":2,
    "viet-nam-nhac-tru-tinh-":3,
    "viet-nam-rap-viet-":4,
    "viet-nam-nhac-thieu-nhi-":5,
    "viet-nam-nhac-cach-mang-":6,
    "viet-nam-nhac-dan-ca-que-huong-":7,
    "viet-nam-nhac-ton-giao-":8,
    "viet-nam-nhac-khong-loi-":9,

    "au-my-classical-":10,
    "au-my-folk-":11,
    "au-my-country-":12,
    "au-my-pop-":13,
    "au-my-rock-":14,
    "au-my-latin-":15,
    "au-my-rap-hip-hop-":16,
    "au-my-alternative-":17,
    "au-my-blues-jazz-":18,
    "au-my-reggae-":19,
    "au-my-r-b-soul-":20,
}

GENRE = "viet-nam-rap-viet-"
FOL_WAV = r"/content/gdrive/MyDrive/KLTN/DatasetWav30s/{}".format(GENRE)
FOL_OUT_MFCC = r"/content/gdrive/MyDrive/KLTN/DatasetMFCCs_20/{}".format(GENRE)

AUDIO_PER_FILE = 500
STATIC_LABEL = obj_genre[GENRE]

if not os.path.exists(FOL_OUT_MFCC):
    os.mkdir(FOL_OUT_MFCC)

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(LAST_HIT,MIN_INDEX_FILE,MAX_INDEX_FILE,num_mfcc=20, n_fft=2048, hop_length=512, num_segments=10):
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
    data = {     
        "labels": [],
        "mfcc": []
    }
    print(str(MIN_INDEX_FILE))
    total_audio = 0
    index_file = MIN_INDEX_FILE
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for f in os.listdir(FOL_WAV):
    # load audio file
        if total_audio >= MIN_INDEX_FILE*AUDIO_PER_FILE and total_audio < MAX_INDEX_FILE*AUDIO_PER_FILE:
            try:
                file_path = os.path.join(FOL_WAV, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T
                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        #print(num_mfcc_vectors_per_segment)
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(STATIC_LABEL)
                        #print("{}, segment:{}".format(file_path, d+1))
            except:
                print("Error")

            if total_audio%AUDIO_PER_FILE==AUDIO_PER_FILE-1:
                print("Dump file :"+str(index_file))
                with open(FOL_OUT_MFCC+"/"+GENRE+str(index_file)+".json", "w+") as fp:
                    json.dump(data, fp, indent=4)
                index_file +=1
                data = {
                    "labels": [],
                    "mfcc": []
                }
        total_audio +=1
    if LAST_HIT==True:
        print("Dump file :"+str(index_file))
        with open(FOL_OUT_MFCC+"/"+GENRE+str(index_file)+".json", "w+") as fp:
            json.dump(data, fp, indent=4)
              
if __name__ == "__main__":
    print(GENRE)
    threads = 4
    t1 = threading.Thread(target=save_mfcc,args=(False,0,2))
    t2 = threading.Thread(target=save_mfcc,args=(False,2,4))
    t3 = threading.Thread(target=save_mfcc,args=(False,4,6))
    t4 = threading.Thread(target=save_mfcc,args=(False,6,8))


    jobs = []

    jobs.append(t1)
    jobs.append(t2)
    jobs.append(t3)
    jobs.append(t4)
    # Start the threads (i.e. calculate the random number lists)
    for j in jobs:
        j.start()

    # Ensure all of the threads have finished
    for j in jobs:
        j.join()
    #save_mfcc(MIN_INDEX_FILE=6,MAX_INDEX_FILE=8,num_segments=10,)
