from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.regularizers import l2
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from prep import dataset
import keras.backend as K
import pandas as pd
import numpy
# user_defined modules
from prep import dataset
IMG_SIZE = 64
class Beng_Train:
    @classmethod
    def build_model(self):
        img_input = Input(shape=(64, 64, 1))
        x = Conv2D(128, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1),
                   padding = 'valid', activation = 'relu',
                   kernel_initializer='he_normal')(img_input)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.5)(x)
        x = LeakyReLU(alpha = 0.05)(x)
        x = Conv2D(64, (3, 3), padding = 'valid',activation = 'relu',
                   kernel_initializer = 'he_normal')(x)
        x = Dropout(0.5)(x)
        x = AveragePooling2D(pool_size=2, strides = None, padding='valid')(x)
        x = Flatten()(x)
        x = Dense(1024, activation = 'relu')(x)

        # multioutput
        dense = Dense(512, activation = 'relu')(x)

        root = Dense(168, activation ='softmax')(dense)
        vowel = Dense(11, activation = 'softmax')(dense)
        consonant = Dense(7, activation = 'softmax')(dense)

        model = Model(inputs = img_input, outputs = [root, vowel, consonant])
        return model
    @classmethod
    def resnet38(self):
        img_input = Input(shape=(64, 64, 1))
        resnet = ResNet50(weights = None, include_top = False,
                          input_shape = (IMG_SIZE, IMG_SIZE, 1))(img_input)
        x = resnet.layers[38].output
        x = Conv2D(128, (3, 3), padding='valid', kernel_initializer='he_normal',
                   activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = Conv2D(64, (3, 3), padding='valid', activation='relu',
                       kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        x = AveragePooling2D(pool_size=2, strides=None, padding='valid')(x)
        x = Flatten()(x)
        # multioutput
        dense = Dense(64, activation='relu')(x)

        root = Dense(168, activation='softmax')(dense)
        vowel = Dense(11, activation='softmax')(dense)
        consonant = Dense(7, activation='softmax')(dense)

        model = Model(inputs=resnet.input, outputs=[root, vowel, consonant])
        return model

    @classmethod
    def train(self, resnet = False):
        if resnet:
            model = self.resnet38()
        else:
            model = self.build_model()
        print(model.summary())
        data, label = dataset('train').get_data()
        Y_train_root = pd.get_dummies(label['grapheme_root']).values
        Y_train_vowel = pd.get_dummies(label['vowel_diacritic']).values
        Y_train_consonant = pd.get_dummies(label['consonant_diacritic']).values

        # Split
        x_train, x_valid, y_train_root,\
        y_valid_root,y_train_vowel, y_valid_vowel,\
        y_train_consonant, y_valid_consonant = train_test_split(data,Y_train_root,
                                                                Y_train_vowel, Y_train_consonant,
                                                                test_size=0.2,
                                                                random_state=666)
        del data
        del label
        del Y_train_root
        del Y_train_vowel
        del Y_train_consonant
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                              metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min',verbose=0, patience=5)
        history = model.fit(x_train, y = {'dense_3':y_train_root,
                                          'dense_4':y_train_vowel,
                                          'dense_5':y_train_consonant},
                            validation_data=(x_valid, [y_valid_root,
                                                       y_valid_vowel,
                                                       y_valid_consonant]),
                            batch_size=256,
                            epochs=10,
                            verbose=1,
                            callbacks=[es])
        del x_train
        del x_valid
        del y_train_root
        del y_valid_root
        del y_train_vowel
        del y_valid_vowel
        del y_train_consonant
        del y_valid_consonant
        return model, history
