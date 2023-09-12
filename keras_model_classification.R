

# The following model is using the libraries keras and tensorflow and the data
# used is the diamonds dataset. Instead of doing a regression-model (can be hard
# because numbers are infinities, also the dataset does not have enough 
# examples), so I chose to turn it into a classification-model. 
# Where I break down the variable price and convert it into factors (classes).
# The model will learn about 15 classes  that are based on the price.

# Loading the libraries
library(tidyverse)
library(keras)
library(tensorflow)

# First, I instantiate the keras library, takes around one minute to instantiate
model <- keras_model_sequential()

# Now reset the  keras  session & set up the random numbers from tensorflow 
tf$keras$backend$clear_session()
tf$keras$utils$set_random_seed(51L)

#Now, we can take a view on values in the diamonds dataset
summary(diamonds)
#Check if the dataset have NAs
sum(is.na(diamonds))
#Check if the dataset have duplicate values
sum(duplicated(diamonds))

# Diamonds dataset have 146 duplicated rows.
# I eliminate duplicates using distinct() and store it in the "data" dataframe.
data <- distinct(diamonds)
# Now, I need to confirm that dataframe "data" doesn't have duplicates
sum(duplicated(data))

# Now, I'll check the price variable and break down into factors (classes).
# For this I will use mutate() to create a new variable called class. And within
# mutate, I'll use cut() that will break down the variable price into factors.
# Each class is in an interval of 1200. And it will be 15 classes.
# Class 0 is from minimum value $326 to the value $1200.
# Class 1 is from value $1201 to the value $2400 and so on, until
# Class 14 is from value $16801 to the max value $18823.
# Also can check the minimum, maximum, quantity and distinct prices by each class.
data |> mutate(class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800, 6000, 
    7200, 8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000),
    labels = c(0:14), include.lowest = TRUE)) |> group_by(class) |> 
    summarise(min_price = min(price), max_price = max(price), quantity = n(), 
    distinct_price = n_distinct(price))

# Now that I've analyzed the variable price, I'll reuse the same code from cut()
# but this time instead of mutate(), I'll use transmute() to just keep
# the variable class that I will use as the *Y-variable(label)*. Then I'll apply 
# Feature Engineering to turn the class variable into one-hot-encoded variable 
# using to_categorical() and then convert the one-hot-encode into a matrix using
# as.matrix() and finally store it in a matrix called ys.
# It will be 15 classes with 15 columns. And all it's one-hot-encoded.
ys <- data |> transmute(class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800,
    6000, 7200, 8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000), 
    labels = c(0:14), include.lowest = TRUE)) |> 
    mutate(class = to_categorical(class)) |> as.matrix()

# Now I'll work on all the *X-variables*. First, eliminate the price variable (
# now the Y-variable is class). Then place the carat variable at the last place.  
# Second, apply feature scaling into all the numeric variables.
# Third, use Feature Engineering to convert the categorical variables cut, color
# and clarity into one-hot-encoded using nested functions. 
# The inner-function as.numeric() executes first
# and converted it to numbers. The outer-function to_categorical() take 
# those values and converted it to one-hot-encoded. Fourth convert the dataframe
# into a matrix using as.matrix() and store it in a matrix called xs.
xs <- data |> select(-price) |> relocate(-carat) |> 
    mutate(across(c(depth, table, x, y, z, carat), ~(scale(.) %>% as.vector)))|> 
    mutate(cut = to_categorical(as.numeric(cut)), 
    color = to_categorical(as.numeric(color)), 
    clarity = to_categorical(as.numeric(clarity))) |> as.matrix()
 
# Now I create a vector of indices that I'll use to split the dataframe
set.seed(51)
index <- sample(x = 3, size = nrow(xs), replace = TRUE, prob = c(.8, .1, .1))

# Check the length for each index
table(index)
# Check the proportions for each index
prop.table(table(index))

# Now I split the dataset xs. I'll use each of the indices over the xs matrix.
# Index 1 is for train set, index 2 for validation set, index 3 for test set
train_x <- xs[index == 1, ]
val_x <- xs[index == 2, ]
test_x <- xs[index == 3, ]

# Now I split the dataset ys. I'll use each of the indices over the ys matrix.
# Index 1 is for train set, index 2 for validation set, index 3 for test set
train_y <- ys[index == 1, ]
val_y <- ys[index == 2, ]
test_y <- ys[index == 3, ]

# Now I build the model with the dense layers and its own activations
model |> layer_dense(units = 512, activation = 'relu'  ,
    input_shape = ncol(train_x))  |>   layer_dense(units = 256, 
    activation = 'relu')  |>   layer_dense(units = 15, activation = 'softmax')

# Now I compile the model with its optimizer, loss and metrics.
model |> compile(loss = loss_categorical_crossentropy(), 
    optimizer = optimizer_adam(), 
    metrics = metric_categorical_accuracy())

# Then I train the model. And use validation_data parameter with the respective 
# validation dataset. Also I'm going to use callback_reduce_lr_on_plateau() in 
# the callback parameter, this help to change the learning_rate when there is no
# improvement in the network. When I run this command on my computer I get:
# For train dataset: loss = .2460; accuracy = .9021
# For validation dataset: loss = .5867; accuracy = .8177
history <- model |> fit(x = train_x, y = train_y, batch_size = 64,epochs = 150, 
    verbose = 1, validation_data = list(val_x, val_y),
    callbacks = callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = .1,
    patience = 25, mode = 'auto', min_lr = .0001))

# Plot the loss and metrics
plot(history)

# Now, I evaluate it on test dataset. And I get: loss = .6067; accuracy = .8170
model |> evaluate(test_x, test_y)

