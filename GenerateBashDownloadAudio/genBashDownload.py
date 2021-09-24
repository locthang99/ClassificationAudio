# CUT 30s to WAV
import os
from os import path
import subprocess
import time
import datetime
import json
#from pydub import AudioSegment
listVN = [
  "viet-nam-cai-luong-",
  "viet-nam-nhac-tre-v-pop-",
  "viet-nam-nhac-trinh-",
  "viet-nam-nhac-tru-tinh-",
  "viet-nam-rap-viet-",
  "viet-nam-nhac-thieu-nhi-",
  "viet-nam-nhac-cach-mang-",
  "viet-nam-nhac-dan-ca-que-huong-",
  "viet-nam-nhac-ton-giao-",
  "viet-nam-nhac-khong-loi-",
]
listAU = [
  "au-my-classical-",
  "au-my-folk-",
  "au-my-country-",
  "au-my-pop-",
  "au-my-rock-",
  "au-my-latin-",
  "au-my-rap-hip-hop-",
  "au-my-alternative-",
  "au-my-blues-jazz-",
  "au-my-reggae-",
  "au-my-r-b-soul-",
]

GENRE = "viet-nam-rap-viet-"
bashPath = r"/content/gdrive/MyDrive/KLTN/ShellBash/BashDown/"+GENRE+".sh"
pathSongJSON = r"/content/gdrive/MyDrive/KLTN/SongJSON/"+GENRE+".txt"
outFol = r"/content/gdrive/MyDrive/KLTN/DatasetSong/"+GENRE
#outFol = r"/content/gdrive/MyDrive/KLTN/DatasetWav30s"

def writeBash(data):
    f = open(bashPath, 'w+')
    f.write(data)
    f.close()

total = 0
count = 0
lineBash = ""
bashAll = ""


def GENBASH(songId, output):
    global count
    global lineBash
    global bashAll
    global total

    #if not os.path.exists(output+"_"+str(part)+".wav"):
    lineBash = lineBash + "curl -s -L api.mp3.zing.vn/api/streaming/audio/"+songId+"/128 -o "+output+"/"+ songId+".mp3 & "

    count += 1
    total += 1
    if count >= 200 and total <= 3000:
        print(total)
        bashAll += "! "+lineBash + "\n"
        lineBash = ""
        count = 0


listSong ={}
def dumpSong():
    with open(pathSongJSON) as fsong:
        lines = fsong.readlines()
    for line in lines:
        try:
            obj = json.loads(line)
            listSong[obj['encodeId']] = obj['duration']
        except:
            pass
    fsong.close()

if __name__ == "__main__":
    for gen in listAU:
        GENRE = gen
        bashPath = r"/content/gdrive/MyDrive/KLTN/ShellBash/BashDown/"+GENRE+".sh"
        pathSongJSON = r"/content/gdrive/MyDrive/KLTN/SongJSON/"+GENRE+".txt"
        outFol = r"/content/gdrive/MyDrive/KLTN/DatasetSong/"+GENRE
        if not os.path.exists(outFol):
            os.mkdir(outFol)
        dumpSong()
        for songId in listSong:
            GENBASH(songId,outFol)
        writeBash(bashAll)
        listSong = {}
        total = 0
        count = 0
        lineBash = ""
        bashAll = ""
    

# for fol in os.listdir(input):
#     if fol == GENRE:
#         pathFol = os.path.join(input, fol)
#         for filename in os.listdir(os.path.join(input, fol)):
#             fullPath = os.path.join(pathFol, filename)
#             # print(fullPath)
#             GENBASH(filename[:-4],fullPath, os.path.join(outFol, fol, filename))

