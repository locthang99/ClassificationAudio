import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.saving.save import load_model
INDEX = 9999
INPUT_MFCC = r""
INPUT_MODEL = r""

mapppingVN = [
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
    "au-my-pop-",
    "au-my-rock-",
    "au-my-latin-",
    "au-my-rap-hip-hop-",
    "au-my-alternative-",
    "au-my-blues-jazz-",
    "au-my-reggae-",
    "au-my-r-b-soul-",
]

def load_data():
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    data = {
        "mfcc": [],
        "labels": []
    }
    label = 0
    for fol in os.listdir(INPUT_MFCC):
        if fol in mapppingVN:
            pathFol = os.path.join(INPUT_MFCC, fol)
            for filename in os.listdir(os.path.join(INPUT_MFCC, fol)):
                check_end_fol = filename[-7:]
                if INDEX >= 10:
                    check_end_fol = filename[-8:]                
                if check_end_fol == "-"+str(INDEX)+".json":
                    fullPath = os.path.join(pathFol, filename)
                    print(fullPath)
                    with open(fullPath, "r") as fp:
                        mfcc_json = json.load(fp)
                    data["mfcc"] += mfcc_json["mfcc"]
                    data["labels"] += [label]*len(mfcc_json["mfcc"])
                    fp.close()
            label += 1

    # with open(data_path, "r") as fp:
    #     data = json.load(fp)
    print("Len mfcc: "+str(len(data["mfcc"])))
    print("Len labels: "+str(len(data["labels"])))

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y
def prepare_data_test():
    # load data
    X, y = load_data()
    X_test = X[..., np.newaxis]
    y_test = y
    return X_test,y_test

def load_model_evaluate():
    # load model
    model = load_model(INPUT_MODEL)

    X_test, y_test = prepare_data_test()
    #model.summary()
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=2)

    print("Loss:")
    print(loss)
    print("Acc:")
    print(accuracy)
    # print("f1_score: ")
    # print(f1_score)
    # print("precision: ")
    # print(precision)
    # print("recal: ")
    # print(recall)



if __name__ == "__main__":
    INDEX = 9
    INPUT_MFCC = r"/content/gdrive/MyDrive/KLTN/DatasetMFCCs_120k"
    INPUT_MODEL = r"/content/gdrive/MyDrive/KLTN/Model_120k/model_40k_Res50_VN_7.h5"
    load_model_evaluate()