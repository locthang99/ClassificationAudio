import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import math
import librosa
from flask import Flask, render_template, request
import requests

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

#HOST_BACKEND = "http://localhost:5000/api/v1/"
HOST_BACKEND = "http://103.92.29.98/api/v1/"
GET_REAL_FILE_BACKEDN_API = HOST_BACKEND + "File/Musics/{}"
GET_TEMP_FILE_BACKEND_API = HOST_BACKEND + "File/Temp?nameFile={}&type={}"
SAVE_FILE_BACKEND_API = HOST_BACKEND + "File/Temp/SaveTempFile"

def toMFCC(file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    list_mfcc = []
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    # load audio file
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(signal,sample_rate)
    part = int((duration - 15)/30)
    print("Duration: "+str(duration))
    # process all segments of audio file
    i = 0
    while i < part:
        mfcc_part =[]
        for d in range(num_segments):
            start = samples_per_segment * (d + i*10)
            finish = start + samples_per_segment
            mfcc = librosa.feature.mfcc(
                signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfcc_part.append(mfcc.tolist())
        # print(np.array(data["mfcc"]))
        x = np.array(mfcc_part)
        x = x[..., np.newaxis]
        list_mfcc.append(x)
        print("save mfcc part "+str(i))
        i+=1
    return list_mfcc

def predict(model, X):
    global t
    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)
    prediction = model.predict(X)
    return prediction[0]


# initalize our flask app
app = Flask(__name__)
model_40k_Res50_VN = load_model("Model/model_40k_Res50_VN_7.h5")
model_40k_Res50_AU = load_model("Model/model_40k_Res50_AU_15.h5")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

mappingTypeVietNam = [
    {'typeId':19,'nametype':'Rap Việt','value':0},
    {'typeId':15,'nametype':'Cải lương','value':0},
    {'typeId':10,'nametype':'Nhạc trẻ V-Pop','value':0},
    {'typeId':14,'nametype':'Nhạc Trịnh','value':0},
    {'typeId':23,'nametype':'Nhạc trữ tình','value':0},
    {'typeId':21,'nametype':'Nhạc thiếu nhi','value':0},
    {'typeId':12,'nametype':'Nhạc cách mạng','value':0},
    {'typeId':13,'nametype':'Nhạc dân ca','value':0},
    {'typeId':16,'nametype':'Nhạc không lời','value':0},
]

mappingTypeAU = [
    {'typeId':30,'nametype':'Classical'},
    {'typeId':31,'nametype':'Folk'},
    {'typeId':32,'nametype':'Country'},
    {'typeId':33,'nametype':'Rock'},
    {'typeId':34,'nametype':'Alternative'},
    {'typeId':35,'nametype':'Rap hip hop'},
    {'typeId':36,'nametype':'Pop'},
    {'typeId':37,'nametype':'Latin'},
    {'typeId':38,'nametype':'Blue jazz'},
    {'typeId':39,'nametype':'R-B Soul'},
    {'typeId':40,'nametype':'Reggae'},
]

@app.route('/')
def index():
    return {'msg':"pong"}

@app.route('/ping/', methods=['GET'])
def testAPI():
    return {'msg':"pong"}

@app.route('/predict_VN/', methods=['GET'])
def predict_VN():
    response = []
    name_file = request.args.get('name_file')
    path_file = GET_TEMP_FILE_BACKEND_API.format(name_file,'audio')
    if not os.path.exists("Input/"+name_file):
        r = requests.get(path_file)
        open("Input/"+name_file,'wb').write(r.content)

    MFCCs = []
    MFCCs=toMFCC("Input"+"/"+name_file)
    
    _part = 0
    for mfcc in MFCCs:
        res_of_part = mappingTypeVietNam[:]
        for segment in mfcc:
            rs_segment = predict(model_40k_Res50_VN, segment)
            for idx, val in enumerate(rs_segment):
                res_of_part[idx]['value'] += val*10
                #res[mappingTypeVietNam[idx]] += val*10

        obj = {'time':str(_part*30+15)+"--"+str(_part*30+45),'predict':res_of_part}
        _part+=1
        response.append(obj)

    result_all = mappingTypeVietNam[:]
    for part in response:
        for idx, res in enumerate(part['predict']):
            result_all[idx]['value'] += res['value']
    response.insert(0,{'time':'All','predict':result_all})
    return({'data':response})

@app.route('/predict_AU/', methods=['GET'])
def predict_AU():
#test predict_AU
    response = []
    name_file = request.args.get('name_file')
    path_file = GET_TEMP_FILE_BACKEND_API.format(name_file,'audio')
    #name_file = (request.query_string).decode("utf-8")
    if not os.path.exists("Input/"+name_file):
        r = requests.get(path_file)
        open("Input/"+name_file,'wb').write(r.content)

    MFCCs = []
    MFCCs=toMFCC("Input"+"/"+name_file)
    
    _part = 0
    for mfcc in MFCCs:
        res_of_part = mappingTypeAU[:]
        for segment in mfcc:
            rs_segment = predict(model_40k_Res50_AU, segment)
            for idx, val in enumerate(rs_segment):
                res_of_part[idx]['value'] += val*10
                #res[mappingTypeVietNam[idx]] += val*10

        obj = {'time':str(_part*30+15)+"--"+str(_part*30+45),'predict':res_of_part}
        _part+=1
        response.append(obj)

    result_all = mappingTypeAU[:]
    for part in response:
        for idx, res in enumerate(part['predict']):
            result_all[idx]['value'] += res['value']
    response.insert(0,{'time':'All','predict':result_all})
    return({'data':response})

@app.route('/real_predict_VN/', methods=['GET'])
def real_predict_VN():
    #real predict_VN
    response = []
    name_file = request.args.get('name_file')
    path_file = GET_REAL_FILE_BACKEDN_API.format(name_file)
    #name_file = (request.query_string).decode("utf-8")
    if not os.path.exists("Input/"+name_file):
        r = requests.get(path_file)
        open("Input/"+name_file,'wb').write(r.content)

    MFCCs = []
    MFCCs=toMFCC("Input"+"/"+name_file)
    
    _part = 0
    for mfcc in MFCCs:
        res_of_part = mappingTypeVietNam[:]
        for segment in mfcc:
            rs_segment = predict(model_40k_Res50_VN, segment)
            for idx, val in enumerate(rs_segment):
                res_of_part[idx]['value'] += val*10
                #res[mappingTypeVietNam[idx]] += val*10

        obj = {'time':str(_part*30+15)+"--"+str(_part*30+45),'predict':res_of_part}
        _part+=1
        response.append(obj)

    result_all = mappingTypeVietNam[:]
    for part in response:
        for idx, res in enumerate(part['predict']):
            result_all[idx]['value'] += res['value']
    response.insert(0,{'time':'All','predict':result_all})
    return({'data':response})

@app.route('/real_predict_AU/', methods=['GET'])
def real_predict_AU():
    #real predict_AU
    response = []
    name_file = request.args.get('name_file')
    path_file = GET_REAL_FILE_BACKEDN_API.format(name_file)
    #name_file = (request.query_string).decode("utf-8")
    if not os.path.exists("Input/"+name_file):
        r = requests.get(path_file)
        open("Input/"+name_file,'wb').write(r.content)

    MFCCs = []
    MFCCs=toMFCC("Input"+"/"+name_file)
    
    _part = 0
    for mfcc in MFCCs:
        res_of_part = mappingTypeAU[:]
        for segment in mfcc:
            rs_segment = predict(model_40k_Res50_AU, segment)
            for idx, val in enumerate(rs_segment):
                res_of_part[idx]['value'] += val*10
                #res[mappingTypeVietNam[idx]] += val*10

        obj = {'time':str(_part*30+15)+"--"+str(_part*30+45),'predict':res_of_part}
        _part+=1
        response.append(obj)

    result_all = mappingTypeAU[:]
    for part in response:
        for idx, res in enumerate(part['predict']):
            result_all[idx]['value'] += res['value']
    response.insert(0,{'time':'All','predict':result_all})
    return({'data':response})

def runLocaltunel():
    os.system('lt --port 8089 --subdomain bigbluebirduit')
def runFlask():
    port = int(os.environ.get('PORT', 8089))   
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    runFlask()
