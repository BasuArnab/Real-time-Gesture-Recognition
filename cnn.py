from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 color_mode='grayscale',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            color_mode='grayscale',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch =9081,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 3632)
from PIL import Image
import numpy as np
from keras.preprocessing import image
def singleImage(name):
    path='dataset/single/'+name+'.png'
    test_image=image.load_img(path,target_size=(64,64))
    test_image = test_image.convert('L')
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    return classifier.predict(test_image)
    
result=singleImage('lol')

model=classifier.to_json()
with open("model.json","w") as f:
    f.write(model)
classifier.save_weights("weight.h5")    
print("Model has been saved !")    
