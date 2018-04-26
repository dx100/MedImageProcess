""" Python kernel for exercise in Lesson 4, transfer learning."""
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

# Compile the model
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
image_size = 224

# The ImageDataGenerator was previously generated with
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# recent changes in keras require that we use the following instead:
data_generator = ImageDataGenerator()

train_generator = data_generator.flow_from_directory(
    '../input/dogs-gone-sideways/images/train',
    target_size=(image_size, image_size),
    batch_size=24,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    '../input/dogs-gone-sideways/images/val',
    target_size=(image_size, image_size),
    class_mode='categorical')


my_new_model.fit_generator(
    train_generator,
    steps_per_epoch=3,
    validation_data=validation_generator,
    validation_steps=1)
