from datetime import datetime

from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from sklearn.model_selection import train_test_split

from build_model import build_unet
from load_data import load_data


def train(X_train, y_train,
          validation_split: float = 0.1,
          batch_size: int = 16,
          epochs: int = 50,
          image_height: int = 128,
          image_width: int = 128,
          image_channels: int = 3) -> History:
    model = build_unet(image_height, image_width, image_channels)
    earlystopper = EarlyStopping(monitor='dice_coef', mode='max', patience=3, verbose=1)
    checkpointer = ModelCheckpoint(f'model/model-{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.h5', verbose=1,
                                   save_best_only=True)
    return model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                     callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3

    SEED = 20
    TEST_RATIO = 0.3

    X, y = load_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=SEED)
    print(len(X_train), len(X_test))
    train(X_train, y_train, image_height=IMG_HEIGHT, image_width=IMG_WIDTH, image_channels=IMG_CHANNELS)
