from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class FruitClassifier:
    def __init__(self):
        self.dataset_labels = np.array(['Ripe', 'Rotten', 'Unripe'])
        # create the base pre-trained model
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 3 classes
        predictions = Dense(2, activation='softmax')(x)  # Update the number of classes here


        self.model = Model(inputs=base_model.input,  outputs=predictions)
        self.model.load_weights(r"C:\Projects\AgriTechiesHackzion\data\ripeness_classification\apple\weights\model_weights.h5") #'../data/ripeness_classification/apple/weights/model_weights.h5')

    
    def predict(self, img_path:str=None, img=None):
        test_image = None
        if img_path == None:
            test_image = img
        else:
            test_image = image.load_img(img_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize the image
        print(type(test_image))
        model_predictions = self.model.predict(test_image)
        predicted_ids = np.argmax(model_predictions, axis=-1)
        predicted_labels = self.dataset_labels[predicted_ids]
        return predicted_labels[0]

clf = FruitClassifier()
print(clf.predict(r"C:\Projects\AgriTechiesHackzion\data\ripeness_classification\apple\train\ripe\ripe_apple_11.png"))
