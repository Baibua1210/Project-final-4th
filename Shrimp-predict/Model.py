import math, json, os, sys


import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
# from PIL import image

# from keras.utils.data_utils import Sequence
# from keras.preprocessing import image


# DATA_DIR = 'data'
TRAIN_DIR = 'C:/Users/pornk/PycharmProjects/Project/testmo/train/'
VALID_DIR = 'C:/Users/pornk/PycharmProjects/Project/testmo/test/'
SIZE = (224, 224)
BATCH_SIZE = 32


if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, samplewise_std_normalization=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, samplewise_std_normalization=True)

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    model = keras.applications.mobilenet_v2.MobileNetV2()


    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=False
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

   # early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('test.h5', verbose=1, save_best_only=True)

    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=100, callbacks=[], validation_data=val_batches, validation_steps=10)
    finetuned_model.save('test_final.h5')