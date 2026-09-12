
# The following network is a time-series model. It uses the dataset sunspots 
# from datasets-package. Also uses the libraries keras3, Tensorflow and Tidyverse.
# The dataset-sunspots is a time-series and the values in data
# are the number of spots in the sun. The values in the dataset are from 
# January-1749 to December-1983. The main-objective of this neural-network
# is if this model can detect and learn patterns in the data and predict the
# sunspots for the month of January-1984

# Loading the libraries
library(keras3)
library(tidyverse)
library(tensorflow, exclude = c("shape", "set_random_seed"))

# Reset the keras session
clear_session(free_memory = TRUE)

# Now set up the random number for all the environment
set_random_seed(51L)

# First is convert the time-series dataset sunspots to an array using the function
# array() and also normalizing the values by 10. Then verify the length of the array
series <- array(sunspots) / 10
length(series)

# Now, define the hyper-parameters to transform the dataset
window_size <- 50

# Split the dataset into train/validation and check the length from both
train_x <- series[1:2000]
val_x <- series[2001:2820]
length(train_x)
length(val_x)

# Now is time to create the sliding windows of data. This is done using the
# function timeseries_dataset_from_array() where the function takes the values 
# from indexes 1 to 50 and group them as the sliding window and also takes the
# value from index 51 and defines it as the label. The next group starts from
# indexes 2 to 51 and the label is the index 52. And so on ...
# This process it is done with the train and validation datasets
train <- timeseries_dataset_from_array(data = train_x, 
    targets = tail(train_x, - window_size), sequence_length = window_size, 
    batch_size = 128)

val <- timeseries_dataset_from_array(data = val_x, targets = tail(val_x, - window_size), 
    sequence_length = window_size, batch_size = 128)

# We can inspect the first batch of data
as_iterator(train) |> iter_next()

# It's time to create the respective inputs and ouputs to build the model. 
 inputs <- keras_input(shape = c(window_size, 1))

outputs <- inputs |> layer_batch_normalization() |> 
    bidirectional(layer = layer_lstm(units = 64, return_sequences = TRUE)) |> 
    bidirectional(layer = layer_lstm(units = 64, return_sequences = TRUE)) |> 
    bidirectional(layer = layer_lstm(units = 32)) |> 
    layer_dense(units = 1)

# Now create the model
model <- keras_model(inputs, outputs)

# Inspect the model's architecture
summary(model)

# Now  compile the model with its optimizer and loss. For the loss I choose to 
# use the  mean-absolute-error
model |> compile(optimizer = optimizer_sgd(), loss = 'mae')

# It's time for training. When the training finished. The metrics are:
# In train dataset the loss = 1.6905; for val dataset the loss = 2.1574
model |> fit(train, epochs = 196, validation_data = val, verbose = 2)

# Next I'm going to extract the last 50 values from the series-object as the 
# last sliding window. Then convert the series to an np.array object with the 
# respective function.
to_predict <- series[2771:2820]
to_predict <- reticulate::np_array(to_predict)

# Now I want to make a prediction that will predicts the sunspots for the entire
# month of January-1984. The prediction is 5.76476. I normalized by 10 so the 
# value predicted of sunpots for the entire month is 57.6476
model |> predict(to_predict [tf$newaxis])

# The real-value of sunspots for January-1984 is: 57.0 
# The model predicted: 57.6476. So the model was capable to detect and learn the 
# patterns in the timeseries.

# To check the real-value we can see it in the dataset sunspot.m2014 
# (its a bigger dataset than sunspots). To can see this dataset need to change 
# some values in R with options() function and the parameter max.print
options(max.print = 3100)
sunspot.m2014

