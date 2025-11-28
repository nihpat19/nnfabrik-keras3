from typing import Dict
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
)
from keras.layers import Concatenate



def unet_single_1024(network_param):
    def local_network_function(input_img):

        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            pool3
        )  # 128 x 128 x 128 (small and thick)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(
            pool4
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        up1 = UpSampling2D((2, 2))(conv5)  # 14 x 14 x 128
        conc_up_1 = Concatenate()([up1, conv4])
        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 64
        conc_up_2 = Concatenate()([up2, conv3])
        conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        up3 = UpSampling2D((2, 2))(conv8)  # 28 x 28 x 64
        conc_up_3 = Concatenate()([up3, conv2])
        conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        up4 = UpSampling2D((2, 2))(conv9)  # 28 x 28 x 64
        conc_up_4 = Concatenate()([up4, conv1])
        conv10 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv10
        )  # 512 x 512 x 1

        return decoded

    return local_network_function

def di_model_function(dataloaders: Dict, seed,**config):
    finetuning = config.get('finetuning',True)
    if finetuning:
        return config['model_path']
    else:
        return unet_single_1024({})