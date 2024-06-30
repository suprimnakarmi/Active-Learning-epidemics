from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet101 

class models():
    def __init__(self, selected_model, img_size):
        self.img_size = img_size

        if selected_model=="vgg16":
            self.feature_dim = 25088
        elif selected_model == "resnet101":
            self.feature_dim = 100352 
        else: 
            self.feature_dim=81536
    
    def vgg(self):
        vgg_pre_t = VGG16(input_shape = (self.img_size, self.img_size, 3),include_top = False, weights ='imagenet')
        return vgg_pre_t

    def resnet(self):
        resnet_pre_t= ResNet101(input_shape = (self.img_size, self.img_size, 3),include_top=False, weights='imagenet')
        return resnet_pre_t

    def densenet(self):
        densenet169_pre_t = DenseNet169(input_shape = (self.img_size, self.img_size, 3),include_top = False, weights ='imagenet' )
        return densenet169_pre_t