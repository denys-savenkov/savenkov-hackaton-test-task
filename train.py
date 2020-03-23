from datetime import datetime

from keras.callbacks import ModelCheckpoint, EarlyStopping, History



def train(model,
          X_train, y_train,
          validation_split: float = 0.1,
          batch_size: int = 16,
          epochs: int = 50) -> History:
    earlystopper = EarlyStopping(monitor='val_dice_coef', mode='max', patience=5, verbose=1)
    checkpointer = ModelCheckpoint(f'model/model-{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.h5', verbose=1,
                                   monitor='val_dice_coef', mode='max', save_best_only=True)
    return model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                     callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    from build_model import build_unet
    from load_data import load_data

    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3

    SEED = 20
    TEST_RATIO = 0.3

    X, y = load_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=SEED)
    print(len(X_train), len(X_test))
    model = build_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    train(model, X_train, y_train)
