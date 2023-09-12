
# The approach of this model is very similar to keras_model_classification.R 
# The main difference is how I apply the Feature_Engineering on the data 
# (one-hot-enconding, normalization, etc) is done with the library tfdataset 
# that is based on Tensorflow-Dataset API. The model will learn about 15 classes  
# that are based on the price.

# Load the libraries
library(tensorflow)
library(keras)
library(tidyverse)
library(tfdatasets)
library(coro)

# Reset the keras session
tf$keras$backend$clear_session()

# Now set up the random number from tensorflow
tf$keras$utils$set_random_seed(51L)

# Select the diamonds dataset and then remove the duplicates 
data <- distinct(diamonds)

# Now apply transformations on the dataset. First relocate the variable carat
# in the last. Second the data-type of the categorical variables cut, color and
# clarity are factors. Tensorflow doesn't recognize this data-type. So, 
# I need to coerce this factors to strings. Third break down the variable price 
# and convert it into classes. Each class is in an interval of 1200. And it will
# be 15 classes. Class 0 is from minimum value $326 to the value $1200.
# Class 1 is from value $1201 to the value $2400 and so on, until
# Class 14 is from value $16801 to the max value $18823. And finally I eliminate
# the variable price, because our variable Y(label) is the variable-class.
data <- data |> relocate(-carat) |>  mutate(cut = as.character(cut), 
    color = as.character(color), clarity = as.character(clarity), 
    class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800, 6000, 7200, 
    8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000), 
    labels = c(0:14), include.lowest = TRUE)) |> 
    mutate(class = to_categorical(class)) |> select(-price)

# Now I create a vector of indices that I'll use to split the dataframe
set.seed(51)
index <- sample(x = 3, size = nrow(data), replace = TRUE, prob = c(.8, .1, .1))

# Check the length for each index
table(index)

# Check the proportions for each index
prop.table(table(index))

# Now I'll split the dataset, I use the index (vectors) to split the dataset.
# Index 1 is for train set, index 2 for validation set, index 3 for test set
train_data <- data[index == 1, ]
val_data <- data[index == 2, ]
test_data <- data[index == 3, ]

# I apply a Feature-Engineering on the data, using the Tensorflow-Dataset API. 
# First create a feature_spec that will contain the transformations and on this 
# step I define the variable-class as the Y (label) and the rest of variables as 
# the X. Second convert the categorical variables cut, color and clarity to 
# one-hot-encoding. Third define the rest of the variables as numeric-columns 
# and then apply a scaler_standar() to normalize the variables. 
spec <- feature_spec(data, class ~ .)
spec <- spec |> step_categorical_column_with_vocabulary_list(cut, 
    vocabulary_list = c('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'), 
    num_oov_buckets = 0L) |> step_indicator_column(cut)
spec <- spec |> step_categorical_column_with_vocabulary_list(color, 
    vocabulary_list = c("D", "E", "F", "G", "H", "I", "J"), 
    num_oov_buckets = 0L) |> step_indicator_column(color)
spec <- spec |> step_categorical_column_with_vocabulary_list(clarity, 
    vocabulary_list = c("I1",   "SI2",  "SI1",  "VS2",  "VS1",  "VVS2", "VVS1", 
    "IF"), num_oov_buckets = 0L) |> step_indicator_column(clarity)
spec <- spec |> step_numeric_column(depth, normalizer_fn = scaler_standard())
spec <- spec |> step_numeric_column(table, normalizer_fn = scaler_standard())
spec <- spec |> step_numeric_column(x, normalizer_fn = scaler_standard())
spec <- spec |> step_numeric_column(y, normalizer_fn = scaler_standard())
spec <- spec |> step_numeric_column(z, normalizer_fn = scaler_standard())
spec <- spec |> step_numeric_column(carat, normalizer_fn = scaler_standard())

# To see the steps in the feature_spec
spec$steps

# Now fit() the feature_spec
spec$fit()

# To see the list of dense-features in the spec
str(spec$dense_features())

# Now, its time to build the model. For this I'm going to use the Functional-API
# Define the inputs of the model and store it in inputs
inputs <- layer_input_from_dataset(train_data %>% select(-class))

# Then define the rest of the model and store it in outputs
outputs <- inputs %>% layer_dense_features(dense_features(spec)) |> 
    layer_dense(units = 512, activation = 'relu') |> 
    layer_dense(units = 256, activation = 'relu') |> 
    layer_dense(units = 15, activation = 'softmax')

# Now define the model with the inputs and outputs
model <- keras_model(inputs = inputs, outputs = outputs)

# Inspect the model's architecture
summary(model)

# Now I compile the model with its optimizer, loss and metrics.
model |> compile(optimizer = optimizer_adam(), 
                 loss = loss_categorical_crossentropy(), 
                 metrics = 'accuracy')

# Then is time for training. For the X, I'll use the dataset and  eliminate the
# variable-class. For the Y define the variable-class as a vector. It applies
# the same approach with the parameter validation_data with their respective
# validation dataset. Also I'm going to use callback_reduce_lr_on_plateau() in 
# the callback parameter, this help to change the learning_rate when there is no
# improvement in the network. When I run this command on my computer I get:
# For train dataset: loss = .2471; accuracy = .9018
# For validation dataset: loss = .6121; accuracy = .8175
history <- model |> fit(x = train_data %>% select(-class), y = train_data$class, 
    epochs = 150, batch_size = 64, 
    validation_data = list(val_data %>% select(-class), val_data$class),
    callbacks = callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = .1,
    patience = 25, mode = 'auto', min_lr = .0001), verbose = 1)

# Plot the loss and metrics
plot(history)

# Now evaluate the model on test dataset. And applies the same approach as 
# in training. I get: loss = .5951; accuracy = .8161
model |> evaluate(test_data %>% select(-class), test_data$class)



