import pandas as pd
import seaborn as sn
from seaborn import matrix
import matplotlib.pyplot as pyplot
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
mapppingVN = [
    "cai-luong",
    "nhac-tre",
    "nhac-trinh",
    "nhac-tru-tinh",
    "rap-viet",
    "nhac-thieu-nhi",
    "nhac-cach-mang",
    "nhac-dan-ca",
    "nhac-khong-loi",
]
mappingAU = [
    "classical",
    "folk",
    "country",
    "pop",
    "rock",
    "latin",
    "rap-hip-hop",
    "alternative",
    "blues-jazz",
    "reggae",
    "r-b-soul",
]
data= [
    [421,5,15,7,2,9,7,33,9],
    [3,379,12,4,40,9,7,8,9],
    [12,6,367,37,7,5,7,8,13],
    [13,17,27,371,6,10,7,8,8],
    [1,33,4,3,412,2,13,15,11],
    [10,13,9,5,18,375,7,14,3],
    [6,14,13,4,8,2,437,11,2],
    [3,12,31,21,14,8,18,373,16],
    [37,6,22,0,25,10,7,13,385],
]
data2= [
    [401,7,13,2,7,10,8,13,8,1,21],
    [15,383,3,6,11,2,36,22,6,2,5],
    [35,6,411,14,9,1,10,3,11,13,6],
    [23,7,3,389,6,5,7,18,8,21,2],
    [10,21,0,13,407,5,4,9,2,14,11],
    [0,4,30,5,38,366,6,21,3,20,1],
    [1,5,3,28,7,12,423,16,2,8,20],
    [12,15,11,6,4,8,18,386,24,12,0],
    [25,16,2,7,4,11,6,5,431,3,7],
    [4,3,23,21,5,5,39,13,12,363,10],
    [17,1,8,14,11,26,3,13,0,10,409],
]
y_true = []
y_pred = []
for idx1, i in enumerate(data2):
    y_true += [idx1+1]*sum(i)
    for idx2, j in enumerate(i):
        y_pred += [idx2+1]*j

sum_true = 0
for idx,i in enumerate(y_true):
    if y_pred[idx] == i:
        sum_true+=1

print(precision_recall_fscore_support(y_true=np.array(y_true),y_pred=np.array(y_pred),average="macro"))
print("acc:")
print(sum_true/len(y_true))
# tp1,fp1,fn1=430,8,70

#normalized_confusion_matrix = data/data.sum(axis = 1, keepdims = True)
df_cm = pd.DataFrame(data2, index = [i for i in mappingAU],
                  columns = [i for i in mappingAU])
pyplot.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,fmt=".4g")
pyplot.show()