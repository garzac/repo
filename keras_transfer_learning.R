# The following keras model its done with Transfer Learning.I used the 
# architecture of mobilenet_v2 and the dataset cifar10. The input_shape of the 
# mobilenet_v2 is 224 * 224 * 3 and the  input_shape of the cifar10 dataset is 
# 32 * 32 * 3. To do this can be done in two ways:
# First, to use the pre-built-model and reduce its input_shape to 32 * 32 *3
# and the cifar10 dataset input_shape also is 32 * 32 * 3, this works but the
# results and the metrics are poor.
# The other way to do this, is to use the pre-built-model with its original
# input_shape of 224 * 224 * 3 and then scale up the input_shape of cifar10 
# dataset from 32 * 32 * 3 and scale it up to 224 * 224 * 3 and in this way the
# results and metrics improves  greatly.

# Load the libraries
library(keras)
library(tidyverse)

# Now load the pre-built-model on mobilenet_v2 architecture and let the original
# input_shape of the model and eliminate the classification-layer
pre_model <- application_mobilenet_v2(input_shape = c(224, 224, 3), 
            include_top = FALSE, weights = 'imagenet')

# Now freeze the layers in the pre_model
pre_model$trainable <- FALSE

# Now I can inspect the pre_model
summary(pre_model)

# Now we create the inputs_shape of the cifar10 dataset and store it in 'inputs'
inputs <- layer_input(shape = c(32, 32, 3))

# I create a new model with the classification-layer and store it in 'outputs'
# In this part, after the 'inputs'-parameter,  I set up the layer-function 
# layer_upsampling_2d(size = c(7, 7)) that this scales up the input_shape of the
# dataset from 32 * 32 * 3 to 224 * 224 * 3
outputs <- inputs |> layer_upsampling_2d(size = c(7, 7)) |> 
    pre_model(training = FALSE) |> layer_global_average_pooling_2d() |> 
    layer_dense(units = 10, activation = 'softmax')

# Now that I have the inputs and the outputs from the model, I define the model
# using keras_model()
model <- keras_model(inputs = inputs, outputs = outputs)

# I can inspect the architecture of the new-model
summary(model)

# Now it's time to get the dataset and split up on train/test  and also x/y
cifar10 <- dataset_cifar10()
train_x <- cifar10$train$x
train_y <- cifar10$train$y
test_x <- cifar10$test$x
test_y <- cifar10$test$y

# Now I delete the dataset cifar10 (to release memory) and then normalize the 
# data from the train_x and test_x
rm(cifar10)
train_x <- train_x / 255
test_x <- test_x / 255

# Now apply Feature-Engineering on the Y-variables train_y and test_y. To do 
# this use to_categorical() to turn the values into one-hot-encode.
test_y <- to_categorical(test_y)
train_y <- to_categorical(train_y)

# Now I compile the model with its optimizer and the loss
model |> compile(loss = loss_categorical_crossentropy(), 
            optimizer = optimizer_adam(), metrics = 'accuracy' )

# Now, all is ready for start the training. Only train the model for 15 epochs,
# because my computer doesn't have a GPU and the training took more than 3 hours
# but I think that the effort of training totally worth it, because improved the 
# results and metrics of the model.
# When the training ended I've got the following metrics.
# For training-set: loss = .6779 | accuracy = .7655
# For test-set: loss = .7670 | accuracy = .7332
# I think that the model still can be improved with more epochs of training.
history <- model |> fit(x = train_x, y = train_y, batch_size = 100, epochs = 15, 
             validation_data = list(test_x, test_y), verbose = 1)

# I can inspect the metrics
plot(history)

# Now apply evaluate() on the model and I get loss = .7669 | accuracy = .7332
model |> evaluate(test_x, test_y)
