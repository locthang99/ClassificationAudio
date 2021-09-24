
import os
INPUT_MFCC = r"/content/gdrive/MyDrive/KLTN/DatasetMFCCs_120k"
INPUT_SONG = r"/content/gdrive/MyDrive/KLTN/DatasetSong"
INPUT_WAV30s = r"/content/gdrive/MyDrive/KLTN/DatasetWav30s"


mapppingVN =[
    "viet-nam-cai-luong-",
    "viet-nam-nhac-tre-v-pop-",
    "viet-nam-nhac-trinh-",
    "viet-nam-nhac-tru-tinh-",
    "viet-nam-rap-viet-",
    "viet-nam-nhac-thieu-nhi-",
    "viet-nam-nhac-cach-mang-",
    "viet-nam-nhac-dan-ca-que-huong-",
    "viet-nam-nhac-khong-loi-",
]

mappingAU = [
  "au-my-classical-",
  "au-my-folk-",
  "au-my-country-",
#   "au-my-pop-",
  "au-my-rock-",
#   "au-my-latin-",
  "au-my-rap-hip-hop-",
  "au-my-alternative-",
#   "au-my-blues-jazz-",
#   "au-my-reggae-",
#   "au-my-r-b-soul-",
]

mappingFull = mapppingVN+mappingAU
GENRE = "au-my-country-"
INPUT = INPUT_MFCC
for fol in os.listdir(INPUT):
    count = 0
    #if fol == GENRE:
    pathFol = os.path.join(INPUT, fol)
    for filename in os.listdir(os.path.join(INPUT, fol)):
        fullPath = os.path.join(pathFol, filename)
        count += 1
    print("Total file in "+pathFol+" :"+str(count))
