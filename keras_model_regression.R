# The following model is a regression-model. And it uses the libraries keras and
# tensorflow. Regression-models can be hard because requires lot of data. I've
# chosen the diamonds dataset and the model it will be predict the price of a 
# diamond based on the variables of the dataset. This model it also uses the 
# library tfdatasets that it is based on the Tensorflow-Dataset API and this
# library process the data and transform it into the required tensors.

# Load the libraries
library(tensorflow)
library(keras)
library(tidyverse)
library(tfdatasets)

# Reset the keras session
tf$keras$backend$clear_session()

# Now set up the random number from tensorflow
tf$keras$utils$set_random_seed(51L)

# Select the diamonds dataset and remove the duplicates 
data <- distinct(diamonds)

# The data-type of the categorical variables cut, color and clarity are factors.
# So, I need to coerce this factors to strings, because Tensorflow doesn't
# recognize this data-type. Also relocate the variables carat and price.
data <- data |> relocate(-c(carat, price)) |> mutate(cut = as.character(cut), 
    color = as.character(color), clarity = as.character(clarity))

# Now, I'm going to extract these indexes that are the row-numbers from the 
# dataset that I'll use for the predictions. Each of these index are based on 
# the quantity of training examples.
indexes <- c(22685, 5462, 3831, 4501, 4038, 11058, 9524, 11220, 11575, 11338, 
    11438, 20066, 8998, 7815, 12089, 10890, 12517, 12551, 14630, 20103,  
    12627, 16222, 11384, 9533, 17489, 17399, 19261, 19268, 16103, 19791)

# Now, I'm going to use the indexes to extract the rows from the dataset and 
# store it in the dataset to_predict 
to_predict <- data[indexes, ]

# Now, eliminate all the row-numbers in the indexes from the dataset data.
data <- data[-indexes, ]

# Verify that none row of the dataset-data exists in the dataset-to_predict
intersect(data, to_predict)

# Extract the variable price as an np.array with the np_array()
price <- reticulate::np_array(data$price)    

# Now, apply a normalization on the array and store it in tensor_price
tensor_price <- layer_normalization(object = price, mean = 2, variance = 1)

# This tensor only have one-dimension and we need to add it another dimension
# to the  tensor and this will give us the required tensor-shape.
tensor_price <- tf$reshape(tensor_price, c(53764L, 1L))

# Finally, eliminate the variable price from the dataset-data, because now, 
# tensor_price will be the Y(label).
data <- data |> select(-price)

# I apply a Feature-Engineering on the data, using the Tensorflow-Dataset API. 
# First create a feature_spec that will contain the transformations. Second 
# convert the categorical variables cut, color and clarity to one-hot-encoding. 
# Third define the rest of the variables as numeric-columns and then apply a 
# scaler_standar() to normalize the variables. 
spec <- feature_spec(dataset = data, ~ .)
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
inputs <- layer_input_from_dataset(data)

# Then define the rest of the model and store it in outputs
outputs <- inputs %>%  layer_dense_features(dense_features(spec)) |> 
    layer_dense(units = 512, activation = 'relu') |> 
    layer_dense(units = 256, activation = 'relu') |> 
    layer_dense(units = 1)

# Now define the model with the inputs and outputs
model <- keras_model(inputs = inputs, outputs = outputs)

# Inspect the model's architecture
summary(model)

# Now compile the model with the respective loss and optimizer. For a regression
# model, the default optimizer is Stochastic-gradient-descent and for the loss
# the default is mean-squared-error but I chose the mean-absolute-error.
model |> compile(optimizer = optimizer_sgd(), loss = loss_mean_absolute_error())

# Now is time to train the model with the respective parameters. On callbacks
# parameter, I chose the function callback_reduce_lr_on_plateau() that will 
# change the learning_rate when there is no improvement in the loss. When the 
# model finished the training, I've get a loss of 212.9998 
# It isn't a good loss, but still it can use other metrics.
model |> fit(x = data, y = tensor_price, batch_size = 128, 
    epochs = 500, verbose = 1, shuffle = TRUE, callbacks = 
    callback_reduce_lr_on_plateau(monitor = 'loss', patience = 25, factor = 0.1,
    mode = 'auto', min_lr = .001))

# Now its time for predictions. I'll use the dataset to_predict and for the
# predictions I'll eliminate the variable-price. And the predictions will be 
# stored as a new-variable called prediction in the dataset to_predict.
to_predict$prediction <- predict(model, to_predict %>% select(-price))

# Now, that I've the prediction of the model and also have the variable price 
# from the dataset. Using these two-variables, I'll use other metric called
# percent_error that measure the difference between the real-value and the
# prediction and is expressed in percentage. And I got the following:
# - Two diamonds are above of 5 percent, one is 6.05%, the other 5.97%
# - Ten diamonds are in the range of 0%
# - Six diamonds are in the range of 1%
# - Seven diamonds are in the range of 2%
# - Five diamonds are in the range of 3%
# I know that based on the loss of this model certainly is not the best model. 
# But also I think that, based on the percent_error, the model it still can
# be useful for persons who have no knowledge in diamonds.
to_predict |> select(-c(x, y , z)) |> 
    mutate(percent_eror = (price - prediction) / price * 100) |> 
    print.data.frame()

