# The following network is a time-series model. It uses the dataset sunspots 
# from datasets-package. Also uses the libraries tensorflow, keras, tfdatasets
# and tidyverse. The dataset-sunspots is a time-series and the values in data
# are the number of spots in the sun. The values in the dataset are from 
# January-1749 to December-1983. The main-objective of this neural-network
# is that the model can be learn patterns in the data and predict the sunspots
# for the month of January-1984

# Load the required libraries
library(keras)
library(tensorflow)
library(tfdatasets)
library(tidyverse)
library(coro)

# First is convert the time-series dataset sunspots to a np.array using 
# the function np_array() from the library reticulate and check the length
series <- reticulate::np_array(sunspots)
length(series)

# Now, define the hyper-parameters to transform the dataset
split_time <- 1600
window_size <- 80
batch_size <- 32
shuffle_buffer <- 1000

# Split the dataset into train/validation and check the length from both
train_x <- series[1:split_time]
val_x <- series[split_time:2819]
length(train_x)
length(val_x)

# Now apply functions to clear previous sessions from tensorflow/keras and set 
# up the seed_numbers for random values
tf$keras$backend$clear_session()
tf$random$set_seed(51L)
set.seed(51)

# Now, I define a function that will transform the dataset. First, convert the
# dataset to slice of tensors. Then will make windows-of-data in the length of
# window_size. Third, apply flat_map() to flatten the data. And then apply
# map() that will split the data into X, Y. For example: X will be the length
# of window_size. Y will be the next-value after window_size. Also apply
# shuffle() and batch() to the data.
window_data <- function(series, window_size, batch_size, shuffle_buffer){
    dataset <- tensor_slices_dataset(series)
    dataset <- dataset_window(dataset = dataset, size = window_size + 1,
                              shift = 1, drop_remainder = TRUE)
    dataset <- dataset_flat_map(dataset, map_func = function(x) 
        dataset_batch(dataset = x, batch_size = window_size + 1))
    dataset <- dataset_map(dataset, map_func = function(window){
        x <- window[1:window_size]
        y <- window[window_size + 1]
        return(c(x, y))
    })
    dataset <- dataset_shuffle(dataset = dataset, buffer_size = shuffle_buffer)
    dataset <- dataset_batch(dataset, batch_size = batch_size)
    return(dataset)
}

# Now, use the function to create the dataset that I'll need to feed the model.
dataset <- window_data(series = train_x, window_size = window_size, 
                       batch_size = batch_size, shuffle_buffer = shuffle_buffer)

# Now, I can inspect the tensors and the shape of tensors in the dataset:
loop(for(x in dataset){
    print(as.list(x))
})

# Define the Neural-Network model and the required layers for this job. I 
# decided to use a combination or layer_lstm() for this particular task.
model <- keras_model_sequential()

model |> layer_lambda(f = function(x) {tf$expand_dims(x, axis = -1L)}, 
                      input_shape = c(NULL)) |> 
    bidirectional(layer = layer_lstm(units = 64, return_sequences = TRUE)) |> 
    bidirectional(layer_lstm(units = 64, return_sequences = TRUE)) |> 
    bidirectional(layer = layer_lstm(units = 32)) |> layer_dense(units = 1) |> 
    layer_lambda(f = function(x) { x * 253.8})

# Now its time to compile the model, the optimizer is SGD() with a specific 
# learning_rate that work well and for the loss is mean-absolute-error.
model |> compile(optimizer = optimizer_sgd(learning_rate = 1e-6), loss = 'mae')

# Now train the model, when trained the mean-absolute-error was 11.6702 
model |> fit(dataset, epochs = 100, verbose = 1)

# Now is time for the prediction of spots in the sun for the month of 
# January-1984. To do this, I select the last 80 values from the series to the
# function predict() that will return a estimated-value of spots in the sun.
# The value that returned is 56.99615
model |> predict(series[2739:2819] [tf$newaxis])

# The real-value of sunspots for January-1984 is: 57.0 
# The model predicted: 56.99615 For me is amazing how the model learned patterns            
# and gave a really close value to the real value.

# To check the real-value we can see it in the dataset sunspot.month 
# (its a bigger dataset than sunspots). To can see this dataset need to change 
# some values in R with options() function and the parameter max.print
options(max.print = 3100)
sunspot.month
