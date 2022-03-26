from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('/home/anondel/brain_tumor/model.h5')

# load the model we saved
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

test_image= image.load_img('/home/anondel/Documents/tumor1.jpg',target_size = (64,64,3))
test_image = image.imgarray(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

print(result)
