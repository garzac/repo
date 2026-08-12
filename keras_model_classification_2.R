
# The following model is using the libraries Keras3 and Tensorflow and the data
# used is the diamonds dataset. Instead of doing a regression-model, I turn it 
# into a classification-model. Where I break down the variable price and convert 
# it into factors (classes). The model will learn about 15 classes  that are 
# based on the variable price. And the model it will learn the class based on 
# the variables of the diamonds dataset.

# Loading the libraries
library(keras3)
library(tidyverse)
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(tfdatasets, exclude = 'shape')

# Reset the keras session
clear_session(free_memory = TRUE)

# Now set up the random number for all the environment
set_random_seed(51L)

# Remove the duplicates and store it in the data-object
data <- distinct(diamonds)

# Now, I'll check the price variable and break down into factors (classes).
# For this I will use mutate() to create a new variable called class. And within
# mutate, I'll use cut() that will break down the variable price into factors.
# Each class is in an interval of 1200. And it will be 15 classes.
# Class 0 is from minimum value $326 to the value $1200.
# Class 1 is from value $1201 to the value $2400 and so on, until
# Class 14 is from value $16801 to the max value $18823.
# Also can check the minimum, maximum, quantity and distinct prices by each class.
# Besides based of the metrics, I notice that all the classes are not balanced.
# And based on that, during training I'll use the appropriate metrics
data |> mutate(class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800, 6000, 
        7200, 8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000),
        labels = c(0:14), include.lowest = TRUE)) |> group_by(class) |> 
    summarise(min_price = min(price), max_price = max(price), quantity = n(), 
              distinct_price = n_distinct(price))

# Now, I'm going to inspect the correlation between all the numeric variables with
# respect to the variable price. And the metric shows that the correlation of the 
# variables depth and table are near to zero. So, both variables it'll discard.
cor(data[, -c(2:4)])

# Now, is time for transform the dataset. First I'm going to relocate the variables
# carat and price as the last variables. Then I'm going to reuse the code to create 
# the variable class and then covert it to an integer-type. Third I'm going to 
# eliminate the variables table, depth and price. Then I'm going to convert the
# categorical variables cut, color and clarity to character-type. And finally
# I will convert the variable class as one-hot-encode
data <- data |> relocate(carat, price, .after = last_col()) |>
    mutate(class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800, 6000, 7200, 
        8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000), 
        labels = c(0:14), include.lowest = TRUE)) |> 
    mutate(class = as.integer(class)) |> select(-c(table, depth, price)) |> 
    mutate(cut = as.character(cut), color = as.character(color), 
        clarity = as.character(clarity)) |> mutate(class = to_categorical(class))


# Now I create a vector of indices that I'll use to split the dataframe
set.seed(51)
index <- sample(x = 3, size = nrow(data), replace = TRUE, prob = c(.7, .15, .15))

# Check the length for each index
table(index)

# Check the proportions for each index
prop.table(table(index))

# Now I'll split the dataset, I use the index (vectors) to split the dataset.
# Index 1 is for train set, index 2 for validation set, index 3 for test set
train_data <- data[index == 1, ]
val_data <- data[index == 2, ]
test_data <- data[index == 3, ]

# It is time to apply some Feature-Engineering on the dataset, using the 
# Tensorflow-Dataset API. First create a feature_space that will contain all the 
# transformations. Second convert the categorical variables cut, color and clarity 
# variables to one-hot-encoding. Third define the rest of the variables as 
# numeric-columns and then in this step also normalize the variables.
# Next, I fit the feature_space using the function adapt() 
feature_space <- layer_feature_space(features = list(
    cut = feature_string_categorical(num_oov_indices = 0, output_mode = 'one_hot'),
    color = feature_string_categorical(num_oov_indices = 0, output_mode = 'one_hot'),
    clarity = feature_string_categorical(num_oov_indices = 0, output_mode = 'one_hot'),
    x = feature_float_normalized(),
    y = feature_float_normalized(),
    z = feature_float_normalized(),
    carat = feature_float_normalized()))

feature_space |> adapt(train_data |> select(-class))


# Then I'm going to extract and the create the respective inputs, features and 
# outputs from feature_space to build a new model
inputs <- feature_space$get_inputs()

features <- feature_space$get_encoded_features()

outputs <- features |> layer_dense(units = 512, activation = 'relu') |> 
    layer_dense(units = 256, activation = 'relu') |> 
    layer_dense(units = 16, activation = 'softmax')

# Now build the model using the respectives inputs and outputs
model <-  keras_model(inputs =  inputs, outputs =  outputs)

# Inspect the model's architecture
summary(model)

# Now I compile the model with its optimizer, loss and metrics. Because the clases
# of the dataset are unbalanced, for the metrics argument I'll use the metrics
# predicion and recall
model |> compile(optimizer = optimizer_adam(), 
                 loss = loss_categorical_crossentropy(), 
                 metrics = c('precision', 'recall'))

# Next, I'm going to build two distinct callbacks that it'll be used during the
# training. The first callback will monitor the LOSS in validation-set and when 
# LOSS stops decreasing, the callback it will change the learning rate. The other
# callback also will monitor the PRECISION in the validation-set and when the 
# PRECISION stops increasing the callback it will stop the training and also 
# it'll RESTORE THE BEST WEIGHTS in the training
lr <- callback_reduce_lr_on_plateau(monitor = 'val_loss',
        patience = 40, verbose = 1, mode = 'min', min_lr = .0001)

stopping <- callback_early_stopping(monitor = 'val_precision', patience = 60, 
        verbose = 1, mode ='max', restore_best_weights = TRUE)

# Now, it's time for training. I'll use the respective data and the callbacks. 
# Training is for 100 epochs, but the training stops in the epoch 61 and the callback
# retrieves the weights from epoch 1. The metrics obtained from training are:
# For train dataset: loss = .8007; precision = .8726; recall = .5998
# For  val  dataset: loss = .5634; precision = .8714; recall = .6970
history <- model |> fit(x = train_data |>  select(-class), y = train_data$class, 
    epochs = 100, batch_size = 128, 
    validation_data = list(val_data |>  select(-class), val_data$class),
    callbacks = list(lr, stopping), verbose = 2)

# Now inspect in test dataset, if the metrics can be similar:
# For test dataset: loss = .6049; precision = .8739; recall = .6914
model |> evaluate(test_data |> select(-class), test_data$class)











