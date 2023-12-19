model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('../data/ripeness_classification/apple/weights/model_weights.h5')
model_predictions = model.predict(test_images)