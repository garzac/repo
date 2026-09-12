
# The following model is a Timeseries model, which uses the libraries Keras3, 
# Tensorflow, Tidyverse and Quantmod. The data which it'll be used in the model 
# are from the Major Stock Markets Indices: Standard & Poors, Dow Jones and Nasdaq. 
# All these indices will be used as the Features(X) in the dataset. And for the 
# data which it'll be considered as the Label(Y) it will be the Oil-Prices.

# The data used corresponds to the period between the date of January 3, 2007 (when 
# Quantmod package starts collecting data) and September 15, 2008 (when Lehman Brothers 
# was declared in bankruptcy). So, the model it'll try to predict the price of Oil on
# September 16, 2008, using the all the Stock Market Indices.
  
# Loading the libraries
library(keras3)
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(tfdatasets, exclude = 'shape')
library(tidyverse)
library(quantmod)

# Obtaining the data using the function getSymbols() from Quantmod package
getSymbols("^IXIC", src = "yahoo")   # Nasdaq Composite
getSymbols("^GSPC", src = "yahoo")   # S&P 500
getSymbols("^DJI", src = "yahoo")    # Dow Jones Industrial Average
getSymbols("DCOILWTICO", src = "FRED") # (WTI) Daily Crude Oil Prices from
                                       # Federal Reserve Bank of St. Louis

# In order to work with this data, I need to apply data wrangling on each of the 
# timeseries. So I need to convert each of the indices to tibbles. Each of the
# timeseries have distinct variables such as Open, High, Low, High, Adjusted and Close.
# I will select the variable Close as the Feature(X) in all the indices. And each
# of the Indices it will be stored in the respective object in form of a tibble.
# The same process it is done in all Indices.
date <- row.names(as.matrix(IXIC))
date <- tibble(date = date)
nasdaq <- as_tibble(IXIC) |> cbind(date) |> relocate(date, .before = 1) |> 
                as_tibble() |> select(date, 'nasdaq_close' = IXIC.Close) 
                

date <- row.names(as.matrix(GSPC))
date <- tibble(date = date)
std <- as_tibble(GSPC) |> cbind(date) |> relocate(date, .before = 1) |> 
                as_tibble() |> select(date, 'std_close' = GSPC.Close) 
                

date <- row.names(as.matrix(DJI))
date <- tibble(date = date)
dow <- as_tibble(DJI) |> cbind(date) |> relocate(date, .before = 1) |> 
                as_tibble() |> select(date, 'dow_close' = DJI.Close)


date <- row.names(as.matrix(DCOILWTICO))
date <- tibble(date = date)
oil <- as_tibble(DCOILWTICO) |> cbind(date) |> relocate(date, .before = 1) |> 
           as_tibble() |> select(date, 'price' = DCOILWTICO) |> 
    mutate(date = as_date(date))


# I want to confirm if all the tibbles have NAs. So I run the function apply()
# to verify if exists NAs by each column. From all the tibbles, the oil dataset
# is the only one that have 373 NAs in the price variable.
apply(is.na(nasdaq), 2, sum)
apply(is.na(dow), 2, sum)
apply(is.na(std), 2, sum)
apply(is.na(oil), 2, sum)


# Now, I'm going to join the three indices into one tibble, the joining will be
# by variable date, besides filter the data from 2007-01-03 to 2008-09-15. And
# it will be stored in the inputs object.
inputs <- inner_join(x = nasdaq, y = dow, by = 'date') |> 
    inner_join(y = std, by = 'date') |> mutate(date = as_date(date)) |> 
    filter(date <= '2008-09-15')


# The oil tibble, have 373 NAs in the variable price. I could discard all the NAs
# in the dataset, but it's common in the commodities prices that can have very
# similar prices over different days, so instead of delete the NAs, I'm going to
# replace the NAs by the price of the previous day. Sometimes the NAs happened in 
# continuous days, so the same will execute the same code twice. Next, I'll filter
# the data on the correct dates and finally I normalize the variable price by 10.
# The object it's stored in the target object.
target <- oil |> mutate(price = case_when(is.na(price) ~ replace(price, 
    values = lag(price)), TRUE ~ price)) |> mutate(price = 
    case_when(is.na(price) ~ replace(price, values = lag(price)), TRUE ~ price)) |> 
    filter(date >= '2007-01-03' & date <= '2008-09-15') |> 
    mutate(price = price / 10)

# I need to compare the dimensions of both tibbles, to verify if have the same
# number of rows. Inputs have 429 rows. Target have 444 rows.
dim(inputs)
dim(target)

# I need to compare the variable date in both datasets, using setdiff() function
# returns all the dates that doesn't match in both datasets. It's stored in the
# object missing_dates
missing_dates <- setdiff(target$date, inputs$date)

# Now that I have all the dates that doesn't match in both datasets, I'm going
# to discard all those dates from the tibble target
target <- target |> filter(! date %in% missing_dates)

# Now, I compare that dates in both datasets are exactly equal using identical().
identical(target$date, inputs$date)

# Now the data is ready. The variable date is not necessary anymore, so I discard 
# the date variable in both datasets. Also the data is converted into matrices.
inputs <- inputs |> select(-date) |> as.matrix()
target <- target |> select(-date) |> as.matrix()


# It's time to split both datasets into train and validation, The first 364 rows
# are for train dataset and the last 65 rows to validation dataset. The same
# process it's done in inputs and target objects. 
train_inputs <- inputs[1:364, ]  
train_target <- target[1:364, ] 

val_inputs <- inputs[365:429, ] 
val_target <- target[365:429, ] 


# Reset the keras session
clear_session(free_memory = TRUE)

# Now set up the random number for all the environment and also I define the 
# hyperparameter window_size
set_random_seed(51L)

window_size <- 30

# Now it's time to create the window-sliding datasets. It works in this way,
# the function takes the first 30 values of each of the three variables from 
# train-inputs and this matrix size (30, 3) it's defined as X and then take the 
# first value of train-target and it's defined as Y. Then takes the values from
# index 2 to 31 from train-inputs as X, with the corresponding index 2 from
# train-target as Y. And so on. This process also is done with validation dataset.
train <- timeseries_dataset_from_array(data = train_inputs, targets = train_target, 
    sequence_length = window_size, batch_size = 128)

val <- timeseries_dataset_from_array(data = val_inputs, targets = val_target, 
    sequence_length = window_size, batch_size = 128)

# We can inspect the first batch of data
as_iterator(train) |> iter_next()

# Next, I'm going to build a callback that it'll be used during the training.
# The callback will monitor the LOSS in the validation-set and when the LOSS
# stops decreasing, the callback it will stop the training and also it'll
# RESTORE THE BEST WEIGHTS in the training
stopping <- callback_early_stopping(monitor = 'val_loss', patience = 35, 
    verbose = 1, mode ='min', restore_best_weights = TRUE)

# It's time to create the respective inputs and ouputs to build the model. 
layer_inputs <- keras_input(shape = c(window_size, 3))

outputs <- layer_inputs |> layer_batch_normalization() |> 
    bidirectional(layer_lstm(units = 32, return_sequences = TRUE)) |> 
    bidirectional(layer_lstm(units = 32, return_sequences = TRUE)) |> 
    bidirectional(layer_lstm(units = 16)) |> 
    layer_dense(units = 1)

# Create the model
model <- keras_model(layer_inputs, outputs)

# We can inspect the model's architecture
summary(model)

# Now  compile the model with its optimizer and loss. For the loss I choose to 
# use the  mean-absolute-error
model |> compile(optimizer = optimizer_sgd(), loss = 'mae')

# It's time for training. The training interrupted in the epoch 193 and restore  
# the weights from epoch 158. The metrics obtained are the following:
# In train dataset the loss = 1.5330; for val dataset the loss = 4.2871
model |> fit(train, epochs = 200, validation_data = val, verbose = 2,
             callbacks = list(stopping))

# Now it's time to make predictions, so I take the last 30 rows from input tibble
# and it's stored in the object to_predict. Also the object it need to be converted
# to a matrix object, and later its converted into a np.array() object
to_predict <- inputs[400:429, ] |> as.matrix()
to_predict <- reticulate::np_array(to_predict)

# Now I make the prediction. The number is 9.124542. I normalized by 10 so the 
# value predicted for the oil price on 16 September, 2008  is: $91.24542
model |> predict(to_predict [tf$newaxis])

# Now, we can inspect the value on that day from the oil tibble. The oil price on 
# September 16, 2008 was: $ 91.50, the difference is almost a quarter of dollar.
# So the model was capable to detect and learn the patterns in the stock indices.
oil |> filter(date == '2008-09-16')

