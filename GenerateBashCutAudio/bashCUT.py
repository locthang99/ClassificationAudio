# CUT 30s to WAV
import os
import subprocess
from sys import path
import time
import datetime
import json
#from pydub import AudioSegment
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

GENRE = "au-my-folk-"
pathSongJSON = r"/content/gdrive/MyDrive/KLTN/SongJSON/"+GENRE+".txt"
input = r"/content/gdrive/MyDrive/KLTN/DatasetSong"
outFol = r"/content/gdrive/MyDrive/KLTN/DatasetWav30s"

if not os.path.exists(outFol+"/"+GENRE):
    os.path.os.mkdir(outFol+"/"+GENRE)

def writeBash(data):
    global index
    f = open("/content/gdrive/MyDrive/KLTN/ShellBash/BashCUT30s/"+GENRE+str(index)+".sh", 'w+')
    f.write(data)
    f.close()

total = 0
count = 0
lineBash = ""
bashAll = ""
index = 100


def CUT(songId,file_path, output):
    #global total
    #dst_wav = output + ".wav"
    #sound = AudioSegment.from_mp3(file_path)
    global count
    global lineBash
    global bashAll
    global total
    global index
    try:
        duration = listSong[songId]
    except:
        duration = 0
    start = 15
    part = 1
    while start+30 < duration:
        begin_cut = str(datetime.timedelta(0, start))
        end_cut = str(datetime.timedelta(0, start+30))
        #print(begin_cut+"  "+end_cut)
        if not os.path.exists(output+"_"+str(part)+".wav"):
            # print(output+"_"+str(part)+".wav")
            lineBash = lineBash + ("ffmpeg -i "+file_path+" -ss "+begin_cut +
                                   " -to "+end_cut+" "+output+"_"+str(part)+".wav -loglevel error & ")
        start += 30
        part += 1
        total += 1
        count += 1
        if count >= 100:
            print(total)
            bashAll += "! "+lineBash + "\n"
            lineBash = ""
            count = 0
    if total >= 4000:
        bashAll += "! "+lineBash + "\n"
        writeBash(bashAll)
        bashAll = ""
        total = 0
        lineBash = ""
        count = 0
        index += 1



listSong ={}
totalSong = 0
def dumpSong():
    global totalSong
    with open(pathSongJSON) as fsong:
        lines = fsong.readlines()
    for line in lines:
        try:
            if totalSong <=3000:
                obj = json.loads(line)
                listSong[obj['encodeId']] = obj['duration']
                totalSong +=1
        except:
            pass


dumpSong()
for fol in os.listdir(input):
    if fol == GENRE:
        pathFol = os.path.join(input, fol)
        for filename in os.listdir(os.path.join(input, fol)):
            fullPath = os.path.join(pathFol, filename)
            # print(fullPath)
            CUT(filename[:-4],fullPath, os.path.join(outFol, fol, filename))
bashAll += "! "+lineBash + "\n"
writeBash(bashAll)
