import json
import os
import math
from re import T
import librosa
import threading

obj_genre = {
    "viet-nam-rap-viet-":4,
    "viet-nam-cai-luong-":0,
    "viet-nam-nhac-tre-v-pop-":1,
    "viet-nam-nhac-trinh-":2,
    "viet-nam-nhac-tru-tinh-":3,
    "viet-nam-nhac-thieu-nhi-":5,
    "viet-nam-nhac-cach-mang-":6,
    "viet-nam-nhac-dan-ca-que-huong-":7,
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

GENRE = ""
FOL_WAV = r"/content/gdrive/MyDrive/KLTN/DatasetWav30s/{}".format(GENRE)
FOL_OUT_MEL = r"/content/gdrive/MyDrive/KLTN/DatasetMelSpectogram/{}".format(GENRE)

AUDIO_PER_FILE = 500
STATIC_LABEL = obj_genre[GENRE]

if not os.path.exists(FOL_OUT_MEL):
    os.mkdir(FOL_OUT_MEL)

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mel(LAST_HIT,MIN_INDEX_FILE,MAX_INDEX_FILE):

    # dictionary to store mapping, labels, and MELs
    data = {     
        "mel": []
    }
    print(str(MIN_INDEX_FILE))
    total_audio = 0
    index_file = MIN_INDEX_FILE

    for f in os.listdir(FOL_WAV):
    # load audio file
        if total_audio >= MIN_INDEX_FILE*AUDIO_PER_FILE and total_audio < MAX_INDEX_FILE*AUDIO_PER_FILE:
            try:
                file_path = os.path.join(FOL_WAV, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                mel = librosa.feature.melspectrogram(signal, sample_rate)
                mel = mel.T
                data["mel"].append(mel.tolist())
            except:
                print("Error")

            if total_audio%AUDIO_PER_FILE==AUDIO_PER_FILE-1:
                print("Dump file :"+str(index_file))
                with open(FOL_OUT_MEL+"/"+GENRE+str(index_file)+".json", "w+") as fp:
                    json.dump(data, fp, indent=4)
                index_file +=1
                data = {
                    "mel": []
                }
        total_audio +=1
    if LAST_HIT==True:
        print("Dump file last hit :"+str(index_file))
        with open(FOL_OUT_MEL+"/"+GENRE+str(index_file)+".json", "w+") as fp:
            json.dump(data, fp, indent=4)
              
if __name__ == "__main__":
    # print(GENRE)
    # save_mel(True,MIN_INDEX_FILE=0,MAX_INDEX_FILE=16)
    print(GENRE)
    threads = 4
    t1 = threading.Thread(target=save_mel,args=(False,0,2))
    t2 = threading.Thread(target=save_mel,args=(False,2,4))
    t3 = threading.Thread(target=save_mel,args=(False,4,6))
    t4 = threading.Thread(target=save_mel,args=(False,6,8))


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
