import cv2 as cv
import tensorflow as tf

from Utils.constants import IMAGE_TO_ELLIPSE_MODEL_PATH


def ai_fit_ellipse_v0(path):
    # read img
    img = cv.imread(path)[:, 304:1744] #Img 1440*1080 since opencv is reversed

    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    model = tf.keras.models.load_model(IMAGE_TO_ELLIPSE_MODEL_PATH + "Classic.h5")

    (height_factor, width_factor) = (img.shape[0]/128, img.shape[1]/128)

    reshaped_gray = cv.resize(gray, (128, 128))

    parameters = model.predict(reshaped_gray.reshape(1, 128, 128)/255.)[0]

    width, height, cent_x, cent_y, angle = parameters
    print("center x,y:", cent_x, cent_y)
    print("diameters:", width, height)
    print("orientation angle:", angle)

    # draw ellipse on input img
    result = cv.resize(img.copy(), (128, 128))
    cv.ellipse(result, (int(cent_x), int(cent_y)), (int(width / 2), int(height / 2)), angle, 0, 360, (0, 0, 255), 2)

    return [result]

