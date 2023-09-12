# This Convolutional Neural Network its done for image-classification.
# The following keras model use the cifar10 dataset. And the model its built
# mostly using the functions layer_conv2d() and layer_max_pooling2d(). 
# Also the model has Data Augmentation that improves the model's metrics.

#First, load the keras and tidyverse libraries
library(keras)
library(tidyverse)
library(tensorflow)

# Now reset the  keras  session & set up the random numbers from tensorflow 
tf$keras$backend$clear_session()
tf$random$set_seed(51L)

# Instantiate the keras library
model <- keras_model_sequential()

# Now load the cifar10 dataset
cifar10 <- dataset_cifar10()

# Now split the dataset into train/test and also x/y
train_x <- cifar10$train$x 
train_y <- cifar10$train$y
test_x <- cifar10$test$x 
test_y <- cifar10$test$y

# Now normalize the matrices train_x/test_y
train_x <- train_x / 255
test_x <- test_x / 255

# Now apply Feature-Engineering on the labels(y) and turn it into one-hot-encode
train_y <- to_categorical(y = train_y, num_classes = 10)
test_y <- to_categorical(y = test_y, num_classes = 10)

# Check the unique one-hot-encode y-variables
unique(train_y)
unique(test_y)

# Now built the Data Augmentation to the dataset
data_augmentation <- FALSE

if (data_augmentation) {
    data_augmentation = keras_model_sequential() %>% 
        layer_random_flip("horizontal") %>% 
        layer_random_rotation(0.2) %>%
        layer_random_height(factor = .2, interpolation = 'nearest') %>%
        layer_random_width(factor = .2, interpolation = 'nearest')
    
    model <- model %>% 
        data_augmentation()
}

# Now build the architecture of the Convolutional Neural Network.
model |> layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = 'same', 
    input_shape = c(32, 32, 3)) |> layer_activation_leaky_relu(.1) |> 
    layer_conv_2d(filters = 32, kernel_size = c(3, 3)) |> 
    layer_activation_leaky_relu(.1) |> 
    layer_max_pooling_2d(pool_size = c(2, 2)) |> layer_dropout(.2) |> 
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = 'same') |> 
    layer_activation_leaky_relu(.1) |> 
    layer_conv_2d(filters = 64, kernel_size = c(3, 3)) |> 
    layer_activation_leaky_relu(.1) |> 
    layer_max_pooling_2d(pool_size = c(2, 2)) |> layer_dropout(.2) |> 
    layer_flatten() |> layer_dense(units = 512) |> 
    layer_activation_leaky_relu(.1) |> layer_dropout(.5) |> 
    layer_dense(units = 10, activation = 'softmax') 

# Inspect the architecture of Convolutional Neural Network
summary(model)

# Compile the Convolutional Neural Network
model |> compile(loss = loss_categorical_crossentropy(), 
    optimizer = optimizer_adam(), 
    metrics = metric_categorical_accuracy())

# Now train the model. I'm going to use callback_reduce_lr_on_plateau() in 
# the callback parameter, this help to change the learning_rate when there is no
# improvement in the network. When the model finished the training, I got:
# For train-dataset: loss = .1741; accuracy = .9377
# For  test-dataset: loss = .6659; accuracy = .8169
history <- model |> fit(x = train_x, y = train_y, batch_size = 100, epochs = 50, 
    verbose = 1, validation_data = list(test_x, test_y), callbacks = 
    callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = .1,
    patience = 5, mode = 'auto', min_lr = .0001))                                                                                                                     

# Inspect the metrics from the model
plot(history)
