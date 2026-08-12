
# The following model is a regression-model. And it uses the libraries Keras3,
# Tensorflow and also uses the Tensorflow-Dataset API that process the data and 
# transform it into the required tensors.  The model it will use the diamonds
# dataset and the model it will be predict the price of a diamond based on 
# all the variables of the dataset. 

# Loading the libraries
library(keras3)
library(tidyverse)
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(tfdatasets, exclude = 'shape')

# Reset the keras session
clear_session()

# Now set up the random number for all the environment
set_random_seed(51L)

# Remove the duplicates and store it in the data-object
data <- distinct(diamonds)

# The data-type of the categorical variables cut, color and clarity are factors.
# So, I need to convert this factors to strings, because Tensorflow doesn't
# recognize this data-type. Also in this step I'm going to normalize the
# variable-Y by dividing the variable-price between 1000
data <- data |> relocate(c(carat, price), .after = last_col()) |> 
    mutate(cut = as.character(cut), color = as.character(color), 
           clarity = as.character(clarity), price = price / 1000)

# Now, I'm going to inspect the correlation between the price variable and the 
# others numeric variables. The variables depth & table have NULL correlation
# with the variable price
cor(data[, c(4:10)])

# Now, I'm going to discard the variables depth & table, because of their NULL
# correlation with the variable price
data <- data |> select(-c(depth, table))

# I'm going to use a subset of the data 
set.seed(51)
sample_data <- data |> sample_n(size = 48000)

# Next, I'll use anti_join() to get the rows of the dataset-data that doesn't
# have a match in sample_data-dataset and store it in the dataset-remnants
remnants <- anti_join(x = data, y = sample_data)

# Verify that none row in both datasets have a match
intersect(x = sample_data, y = remnants)

# Now, I'm going to create a vector of indexes that later I'll be using for 
# making predictions 
set.seed(51)
index_topredict <- sample(x = nrow(sample_data), size = 150, replace = FALSE)

# Now, I'm going to use the vector indexes to extract the rows from the dataset 
# and store it in the dataset to_predict 
to_predict <- sample_data[index_topredict, ]

# Now, eliminate all the row-numbers in the indexes from the dataset data.
sample_data <- sample_data[-index_topredict, ]

# Now I'm going to create another vector of indices that I'll use to split the 
# dataset-sample_data into three subsets of data for train/validation/test
set.seed(51)
index_tosplit <- sample(x = 3, size = nrow(sample_data), replace = TRUE,
                        prob = c(.7, .15, .15))

# Check the length for each index
table(index_tosplit)
# Check the proportions for each index
prop.table(table(index_tosplit))

# Now, it is time to split the data into subsets of data
# Index 1 is for training-set, index 2 for validation-set, index 3 for test-set
train <- sample_data[index_tosplit == 1, ]
val <- sample_data[index_tosplit == 2, ]
test <- sample_data[index_tosplit == 3, ]

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

# Then I'm going to extract and the create the respective inputs, features and 
# outputs from feature_space to build a new model
feature_space |> adapt(data |> select(-price))

inputs <- feature_space$get_inputs()

features <- feature_space$get_encoded_features()

outputs <- features |> layer_dense(units = 1024, activation = 'relu') |> 
    layer_dense(units = 512, activation = 'relu') |> 
    layer_dense(units = 1)

# Now build the model using the respectives inputs and outputs
model <-  keras_model(inputs =  inputs, outputs =  outputs)

# Inspect the model's architecture
summary(model)

# Next, I'm going to build two distinct callbacks that it'll be used during the
# training. The first callback will monitor the MAE in validation-set and when the
# MAE stops decreasing, the callback it will change the learning rate. The other
# callback also will monitor the MAE in the validation-set and when the MAE
# stops decreasing the callback it will stop the training and also it'll RESTORE
# THE BEST WEIGHTS in the training
lr <- callback_reduce_lr_on_plateau(monitor = 'val_mae',
        patience = 35, verbose = 1, mode = 'min', min_lr = .001)

stopping <- callback_early_stopping(monitor = 'val_mae', patience = 50, 
        verbose = 1, mode ='min', restore_best_weights = TRUE)

# Now compile the model with the respective loss and optimizer. For a regression
# model, the default optimizer is Stochastic-gradient-descent and for the loss
# is mean-squared-error. Also in metrics I'll use the mean-absolute-error 
model |> compile(optimizer = optimizer_sgd(), 
                 loss = loss_mean_squared_error(), metrics = 'mae')

# Then it is time for training. Also use the parameter validation_data with the 
# repective validation dataset. Also I'll use both callbacks in the callback
# parameter, and both callbacks helps to improve the model. When I ran it in my
# computer the training stopped in the epoch 484 and retrieves the weights from
# the epoch 434. The values obtained are the following:
# For train dataset: loss = .2237;  mae = .2465  
# For validation dataset: loss = .2783;  mae = .2716 
# When I normalized the variable-price by dividing it between $1000 and if the
# mae is .2716 in the validation dataset; this means that the predictions  
# are off by $271 on average
history <- model |> fit(x = train |> select(-price), y = train$price,  
    batch_size = 128,  epochs = 500, verbose = 2, shuffle = TRUE, 
    callbacks = list(lr, stopping) , 
    validation_data = list(val |> select(-price), val$price))

# It's time to evaluate the test dataset. The loss = .2838; mae = .2709
model |> evaluate(test |> select(-price), test$price)

# Now its time for predictions. I'll use the dataset to_predict and for the
# predictions I'll eliminate the variable-price. And the predictions will be 
# stored as a new-variable called prediction in the dataset to_predict.
to_predict$prediction <- predict(model, to_predict |> select(-price))

# It's time to evaluate the predictions. I normalized the variable-price by 
# dividing between 1000, so the predictions can be in the same scale, I'm going
# to multiply the variable PREDICTION & PRICE per 1000. Then I'll create the variable
# PERCENT_ERROR to evaluate the difference-in-percent between the prediction and 
# the price. Also I'll create another variable PERCENT that will convert the 
# PERCENT_ERROR into an integer with absolute-value. Then I'll use count() to 
# count the number of observations of each value in the variable PERCENT. Then 
# I'll create the variable CUM_SUM that will realize cumulative-sum of the number 
# of observations. Finally I'll create the variable RATIO that will calculate the 
#percentage of each value in the variable PERCENT by dividing CUM_SUM by the total.
# The ratio of each value in the variable PERCENT are:
# From 0% to 3% the ratio is 42%; from 5% to below the ratio is 60%
# From 7% to below the ratio is 73%; from 10% to below the ratio is 85%
# This means that 85% of the predictions in the dataset,  the PERCENT_ERROR
# is less or equal to 10% 
to_predict |> mutate(prediction = prediction * 1000, price = price * 1000) |> 
    mutate(percent_eror = (price - prediction) / price * 100) |> 
    mutate(percent = as.integer(abs(percent_eror))) |> count(percent) |> 
    mutate(cum_sum = cumsum(n)) |> mutate(ratio = round(cum_sum / 150 * 100, 2)) |> 
    print.data.frame()

# I'm going to reuse the code from PREDICTION. So, I'm going to multiply the 
# variable PREDICTION & PRICE per 1000. Then I'll create the variable DIFFERENCE  
# that will calculate the difference between the prediction and the price. And the  
# value will be rounded with absolute value. Also I'll apply the function quantile()
# to the variable DIFFERENCE to calculate the quartiles from 5%, 10%, 15% until 100%. 
# And I get the next values:
# The 25% quartile is $29; the median is $68; the 75% quartile is $227.25 
# The 85% quartile is $436.80; this means that the 85% of the values of DIFFERENCE 
# between the prediction and the price are less or equal to $436.80
to_predict |> mutate(prediction = prediction * 1000, price = price * 1000) |> 
    mutate(difference = round(abs(price - prediction))) |> 
    pull(difference) |> quantile(seq(.05, 1, .05))

# I'm going to reuse the code with the variables PREDICTION and DIFFERENCE from the 
# previous code. Then I want to group by the variable CUT. Next, I want to apply
# the quartile() to the variable DIFFERENCE to get the different quartiles of
# the variable, but the code will return multiple outputs by each observation.
# And I'll need to apply data-wrangling to the data, so the data format can be 
# readable. So, also apply the function pivot_wider() and unnest() to achieve it
to_predict |> mutate(prediction = prediction * 1000, price = price * 1000) |> 
    mutate(difference = round(abs(price - prediction)) ) |> 
    group_by(cut) |> reframe(quartile = quantile(difference)) |> 
    pivot_wider(names_from = cut, values_from = quartile, values_fn = list) |> 
    unnest(cols = c('Fair', 'Good', 'Ideal', 'Premium', 'Very Good')) |> 
    add_column('quartile' = c('0%', '25%', '50%', '75%', '100%'), .before = 1)

# Let's try if with a much bigger dataset, I can get similar metrics.
# So, I get a new tibble with 3000 rows with new data from the remnants-dataset
set.seed(51)
big_prediction <- remnants |> sample_n(size = 3000)

# Also apply the predictions for the new dataset
big_prediction$prediction <- predict(model, big_prediction |> select(-price))

# Same as above. I'm going to reuse the code from PREDICTION. So, I'm going to 
# multiply the variable PREDICTION & PRICE per 1000. Then I'll create the variable
# PERCENT_ERROR to evaluate the difference-in-percent between the prediction and 
# the price. Also I'll create another variable PERCENT that will convert the 
# PERCENT_ERROR into an integer with absolute-value. Then I'll use count() to 
# count the number of observations of each value in the variable PERCENT. Then 
# I'll create the variable CUM_SUM that will realize cumulative-sum of the number 
# of observations. Finally I'll create the variable RATIO that will calculate the 
# percentage of each value in the variable PERCENT by dividing CUM_SUM by the total.
# The ratio of each value in the variable PERCENT are:
# From 0% to 3% the ratio is 40%; from 5% to below the ratio is 55%
# From 7% to below the ratio is 68%; from 10% to below the ratio is 81%
# This means that 81% of the predictions in the dataset,  the PERCENT_ERROR
# is less or equal to 10%  
big_prediction |> mutate(prediction = prediction * 1000, price = price * 1000) |> 
    mutate(percent_eror = abs(price - prediction) / price * 100) |> 
    mutate(percent = as.integer(percent_eror)) |> count(percent) |> 
    mutate(sums = cumsum(n))|> mutate(ratio = round(sums / 3000 * 100, 2)) |> 
    print.data.frame()

# Same as above, I'm going to reuse the code from PREDICTION. So, I'm going to 
# multiply the variables PREDICTION & PRICE per 1000. Then I'll create the variable
# DIFFERENCE that will calculate the difference between the prediction and the
# price. And the value will be rounded with absolute value.
# Also I'll apply the function quantile() to the variable DIFFERENCE to calculate
# the quartiles from 5%, 10%, 15% until 100%. And I get the next values:
# The 25% quartile is $41; the median is $106; the 75% quartile is $295 
# The 85% quartile is $498.15; this means that the 85% of the values of DIFFERENCE 
# between the predictions and the price are less or equal to $498.15
big_prediction |> mutate(prediction = prediction * 1000, price = price  * 1000) |> 
    mutate(difference = round(abs(price - prediction))) |> 
    pull(difference) |> quantile(seq(.05, 1, .05))

# I'm going to reuse the code with the variables PREDICTION and DIFFERENCE from the
# previous code. Then I want to group by the variable CUT. Next, I want to apply
# the quartile() to the variable DIFFERENCE to get the different quartiles of
# the variable, but the code will return multiple outputs by each observation.
# And I'll need to apply data-wrangling to the data, so the data format can be 
# readable. So, also apply the function pivot_wider() and unnest() to achieve it
big_prediction |> mutate(prediction = prediction * 1000, price = price * 1000) |> 
    mutate(percent_eror = abs(price - prediction) / price * 100) |> 
    mutate(difference = round(abs(price - prediction)))  |> 
    group_by(cut) |> reframe(quantile = quantile(difference)) |> 
    pivot_wider(names_from = cut, values_from = quantile, values_fn = list) |> 
    unnest(cols = c('Fair', 'Good', 'Ideal', 'Premium', 'Very Good')) |> 
    add_column('quartile' = c('0%', '25%', '50%', '75%', '100%'), .before = 1)

# I'm going to reuse the variables PREDICTION, PERCENT_ERROR, PERCENT, DIFFERENCE.
# The code will be similar as the previous code, I'm going to apply the quantile()
# to the variable DIFFERENCE to get the different quartiles of the variable, but 
# this time I'm going to group by the variable PERCENT, and the code will return 
# multiple outputs by each observation. And I'll need to apply data-wrangling to 
# the data, so the data format can be readable. But this time, I'll need to apply 
# distinct functions such pivot_wider(), unnest() and pivot_longer() to achieve it
big_prediction |> mutate(prediction = prediction * 1000, price = price * 1000) |> 
    mutate(percent_eror = (price - prediction) / price * 100) |> 
    mutate(percent = abs(as.integer(percent_eror))) |> 
    mutate(difference = round(abs(price - prediction)) ) |> 
    group_by(percent) |> reframe(quant = quantile(difference)) |> 
    pivot_wider(names_from = percent, names_prefix = 'percent_', 
        values_from = quant, values_fn = list) |> 
    unnest(cols = c(percent_0, percent_1, percent_2, percent_3, percent_4, 
        percent_5, percent_6, percent_7, percent_8, percent_9, percent_10, 
        percent_11, percent_12, percent_13, percent_14, percent_15, percent_16, 
        percent_17, percent_18, percent_19, percent_20, percent_21, percent_22,
        percent_23, percent_24, percent_25, percent_26, percent_27, percent_28, 
        percent_30, percent_31, percent_32, percent_33, percent_34, percent_35, 
        percent_36, percent_37, percent_38, percent_39, percent_43, percent_44, 
        percent_45, percent_47, percent_50, percent_51, percent_54, percent_68)) |> 
    add_column('quartile' = c('0%', '25%', '50%', '75%', '100%'), .before = 1) |> 
    pivot_longer(cols = -quartile, names_to = 'percent', values_to = 'values') |> 
    pivot_wider(names_from = quartile, names_prefix = 'quartile_' , 
                values_from = values) |> print.data.frame()

