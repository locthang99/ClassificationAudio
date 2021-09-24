import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

# Inherit Layer and establish a resnet50 101 152 convolutional layer module
class CellBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(CellBlock, self).__init__()

        self.conv1 = Conv2D(filter_num[0], (1,1), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.conv2 = Conv2D(filter_num[1], (3,3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')

        self.conv3 = Conv2D(filter_num[2], (1,1), strides=1, padding='same')
        self.bn3 = BatchNormalization()

        self.residual = Conv2D(filter_num[2], (1,1), strides=stride, padding='same')
        
    def call (self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        r = self.residual(inputs)

        x = layers.add([x, r])
        output = tf.nn.relu(x)

        return output

#Inherit the Model, create resnet50 101 152
class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            Conv2D(64, (7,7), strides=(2,2),padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3,3), strides=(2,2), padding='same')
        ]) #Start module

        # Number of filters in different convolutional layers
        filter_block1=[64, 64, 256]
        filter_block2=[128,128,512]
        filter_block3=[256,256,1024]
        filter_block4=[512,512,2048]

        self.layer1 = self.build_cellblock(filter_block1, layers_dims[0]) 
        self.layer2 = self.build_cellblock(filter_block2, layers_dims[1], stride=2)
        self.layer3 = self.build_cellblock(filter_block3, layers_dims[2], stride=2)
        self.layer4 = self.build_cellblock(filter_block4, layers_dims[3], stride=2)

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(nb_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x=self.stem(inputs)
        # print(x.shape)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        
        x=self.avgpool(x)
        x=self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, stride)) #The first block stride of each layer may be non-1

        for _ in range(1, blocks):      #How many blocks each layer consists of
            res_blocks.add(CellBlock(filter_num, stride=1))

        return res_blocks


def build_ResNet(NetName, nb_classes):
    ResNet_Config = {'ResNet50':[3,4,6,3],
                    'ResNet101':[3,4,23,3],
                    'ResNet152':[3,8,36,3]}

    model= ResNet(ResNet_Config[NetName], nb_classes)
    model.build(input_shape=(None, 130, 13, 1))
    return model

def main():
    model = build_ResNet('ResNet50', 9)
    model.summary()

if __name__=='__main__':
    main()
    
