def resnet50_classify(img_path, top=3):
    import keras
    from keras.applications.resnet50 import ResNet50
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np

    model = ResNet50(weights='imagenet')

    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    return decode_predictions(preds, top=top)[0]

def xception_classify(img_path, top=3):
    import keras
    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input, decode_predictions
    import numpy as np

    model = Xception(weights='imagenet')

    img = keras.utils.load_img(img_path, target_size=(299, 299))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    return decode_predictions(preds, top=top)[0]