
from Model.FCN import FCN
# from Model.Res18 import build_ResNet
# from Model.Res50 import ResNet50
import keras

model = FCN()
model.summary()
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,rankdir='TB', expand_nested=False, dpi=96)