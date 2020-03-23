import glob
import os

from keras.engine.saving import load_model

from dice_loss import dice_coef, dice_loss


def load_last_model():
    list_of_files = glob.glob('model/*.h5')
    latest_file = max(list_of_files, key=os.path.getctime)
    return load_model(latest_file, custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss})
