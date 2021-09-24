import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os

from tensorflow.python.keras.saving.save import load_model





INDEX = 0
INPUT_MEL = r"/content/gdrive/MyDrive/KLTN/DatasetMelSpectogram"
OUTPUT_MODEL = ""
BEFORE_MODEL = ""

mappingVN = [
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

mappingFull = mappingVN+mappingAU


def load_data():
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    data = {
        "mel": [],
        "labels": []
    }
    label = 0
    for fol in os.listdir(INPUT_MEL):
        if fol in mappingVN:
            pathFol = os.path.join(INPUT_MEL, fol)
            for filename in os.listdir(os.path.join(INPUT_MEL, fol)):
                check_end_fol = filename[-7:]
                if INDEX >= 10:
                    check_end_fol = filename[-8:]                
                if check_end_fol == "-"+str(INDEX)+".json":
                    fullPath = os.path.join(pathFol, filename)
                    print(fullPath)
                    with open(fullPath, "r") as fp:
                        mel_json = json.load(fp)
                    data["mel"] += mel_json["mel"]
                    if fol in mappingVN:
                      label =0
                      data["labels"] += [label]*len(mel_json["mel"])
                    else:
                      label =1
                      data["labels"] += [label]*len(mel_json["mel"])
                    fp.close()

    # with open(data_path, "r") as fp:
    #     data = json.load(fp)
    print("Len mel: "+str(len(data["mel"])))
    print("Len labels: "+str(len(data["labels"])))

    X = np.array(data["mel"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data()

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


def train():
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2)

    # # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    if INDEX == 0:
        model = ResNet50()
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    else:
        print(BEFORE_MODEL)
        model = load_model(BEFORE_MODEL)

    # # compile model

    #model.compile(optimizer=optimiser,
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy',f1_m,precision_m, recall_m])

    #model.summary()

    # # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # plot accuracy/error for training and validation
    plot_history(history)
    model.save(OUTPUT_MODEL)
    # # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)



if __name__ == "__main__":

    INDEX = 0
    while INDEX < 1:
        OUTPUT_MODEL = r"/content/gdrive/MyDrive/KLTN/Model_120k/model_40k_Res50_VN_Mel_" + str(INDEX)+".h5"
        BEFORE_MODEL = r"/content/gdrive/MyDrive/KLTN/Model_120k/model_40k_Res50_VN_Mel_" + str(INDEX-1)+".h5"
        train()
        INDEX += 1



    INDEX = 4
    OUTPUT_MODEL = r"/content/gdrive/MyDrive/KLTN/Model_120k/model_40k_Res50_VN_Mel_" + str(INDEX)+".h5"
    BEFORE_MODEL = r"/content/gdrive/MyDrive/KLTN/Model_120k/model_40k_Res50_VN_Mel_" + str(2)+".h5"
    train()
    # INDEX = 5
    # while INDEX < 10:
    #    OUTPUT_MODEL = r"/content/gdrive/MyDrive/KLTN/Model_120k/model_40k_Res50_VN_Mel_" + str(INDEX)+".h5"
    #    BEFORE_MODEL = r"/content/gdrive/MyDrive/KLTN/Model_120k/model_40k_Res50_VN_Mel_" + str(INDEX-1)+".h5"
    #    train()
    #    INDEX += 1