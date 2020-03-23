def predict_masks(model, X_test):
    return model.predict(X_test, verbose=1)


if __name__ == '__main__':
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imshow
    from sklearn.model_selection import train_test_split

    from load_data import load_data
    from utils import load_last_model

    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3

    SEED = 20
    TEST_RATIO = 0.3

    model = load_last_model()
    X, y = load_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=SEED)
    y_pred = predict_masks(model, X_test)

    ix = random.randint(0, len(y_pred))
    imshow(X_test[ix])
    plt.show()
    imshow(np.squeeze(y_test[ix]))
    plt.show()
    imshow(np.squeeze(y_pred[ix]))
    plt.show()
