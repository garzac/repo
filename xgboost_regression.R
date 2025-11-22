
# The following model is a regression-model using the algorithm XGBoost.
# The model will use the diamonds dataset. And the model it will predict the price
# of the diamonds based on all the variables of the dataset.

# Loading the required libraries 
library(tidyverse)
library(xgboost)

# Removing the duplicated data
data <- distinct(diamonds)

# The data type of the categorical variables cut, color and clarity are factors.
# So, I need the convert it to a numeric data type
data <- data |> mutate(across(where(is.factor), as.numeric))

# Now, I'm going to create a vector of indexes that later I'll be using for 
# making predictions. The size of the dataframe is 3000 rows. 
set.seed(51)
index_to_predict <- sample(x = nrow(data), size = 3000, replace = FALSE)

# Now, I'm going to use the vector indexes to extract the rows from the dataset 
# and store it in the dataset to_predict
to_predict <- data[index_to_predict, ]
# Now, eliminate all the row-numbers in the indexes from the dataset data.
data <- data[-index_to_predict, ]
# Verify that none row of the dataset data exists in the dataset to_predict
intersect(data, to_predict)
# Now, convert the dataset to_predict to a matrix
to_predict <- to_predict |> as.matrix()

# Now, I'm going to apply normalization in the variable price using scale()
data <- data |> mutate(price = scale(price))
# This time, I'm going to extract the metrics MEAN & STD DEV from the variable
# price. Later, I will use both metrics to perform calculations
original_mean <- attr(data$price, "scaled:center")
original_sd <- attr(data$price, "scaled:scale")

# Now I'm going to create another vector of indices that I'll use to split the 
# dataset data into two subsets of data for training and testing
set.seed(51)
index <- sample(x = 1:2, size = nrow(data), replace = TRUE, prob = c(.9, .1))
# Check the proportions for each index
prop.table(table(index))

# Now, is time to split the dataset into subsets of data.
# Index 1 is for training, Index 2 is for testing
train <- data[index == 1, ]
test <- data[index == 2, ]

# So the data can work in the XGBoost algorithm, the data it will need to be 
# transformed. First remove the variable-Y (price) from the dataset. Then convert 
# it to a matrix. And finally do the conversion to a xgb.DMatrix object and in 
# this step, I define the variable-price as the label. This process need to be 
# done in both datasets
train <- train |> select(-price) |> as.matrix() |> xgb.DMatrix(label = train$price) 
test <- test |> select(-price) |> as.matrix() |> xgb.DMatrix(label = test$price)

# To evaluate both datasets train and test in the training process, the parameter
# WATCHLIST will be used, and for this I'll create a list-object with both datasets
watchlist <- list(train = train, test= test)

# Is time to train the model. I set up the parameter WATCHLIST to can inspect the 
# metrics in both datasets. And the object is stored in model 
# When the training is completed the metrics are: 
# for train-set the loss is: 0.052679 and for the test-set the loss is: 0.138867
model <- xgb.train(data = train, objective = 'reg:squarederror', nrounds = 550, 
            watchlist = watchlist, verbose = 1, print_every_n = 10)

# Now is time to make the predictions with the new data to_predict. When I do
# the predictions, I eliminate the variable price that is the column number 7
# and the predictions are stored in the object-prediction
prediction <- predict(model, to_predict[, -7])

# Now is time to evaluate the predictions, but first I need to apply data-wrangling.  
# The object prediction is a vector and the first thing is convert it to a tibble.
# Then I merge the object prediction with the object to_predict using cbind() and 
# then convert it to a tibble-object using as_tibble(). Last relocate the variables 
# price & prediction at the end of the tibble and then store it in the object df
df <- data.frame(prediction = prediction) |> cbind(to_predict) |> as_tibble() |> 
    relocate(price, prediction, .after = last_col())

# When I normalized the variable-price, I extracted the metrics MEAN and STD DEV.
# The predictions were made on the same scale that the variable-Y was normalized.
# But in the dataframe to_predict, I have the real values of the diamonds, that
# are stored in the variable-price, so I can compare the predictions with the
# real values. To do this I need to apply inverse-scaling using the metrics
# MEAN and STD DEV, and doing this, all the predictions will be on the same scale 
# that the real values (stored in the variable-price).

# I create the variable UNSCALED_PRICE in which I apply the inverse-scaling of the 
# predictions, the formula is: PREDICTION * STD DEV + MEAN. Then also create the
# variable PERCENT_ERROR to evaluate the difference-in-percent between the 
# prediction and the price. Also create another variable PERCENT that will convert
# the variable PERCENT_ERROR into an integer with absolute-value. Then I'll use
# count() to count the number of observations of each value in the variable PERCENT.
# Then I create the variable CUM_SUM that will realize a cumulative-sum of the 
# number of observations. Finally I'll create the variable RATIO that will 
# calculate the percentage of each value in the variable PERCENT by dividing 
# CUM_SUM by the total. The ratio of each value in the variable PERCENT are:
# From 0% to 3% the ratio is 43%; from 5% to below the ratio is 58%
# From 7% to below the ratio is 69%; from 10% to below the ratio is 82%
# This means that 82% of the predictions in the dataset,  the PERCENT_ERROR
# is less or equal to 10% 
df |> select(-c(x, y, z,  table, depth)) |> 
    mutate(unscaled_price = prediction * original_sd + original_mean) |> 
    mutate(percent_error = (price - unscaled_price) / price * 100) |>
    mutate(percent = as.integer(abs(percent_error))) |> 
    count(percent) |> mutate(cum_sum = cumsum(n)) |> 
    mutate(ratio = round(cum_sum / 3000 * 100, 2)) |> print.data.frame()

# In this evaluation, I'm going to reuse the variable UNSCALED_PRICE. Then I'll 
# create the variable DIFFERENCE that will calculate the difference between the
# prediction and the price. And the value will be rounded with absolute-value.
# Also I'll apply the function quantile() to the variable DIFFERENCE to calculate
# the quartiles from 5%, 10%, 15% until 100%. And I get the next values:
# The 25% quartile is $35; the median is $104; the 75% quartile is $314 
# The 85% quartile is $517.15; this means that the 85% of the values of DIFFERENCE 
# between the prediction and the price are less or equal to $517.15
df |> select(-c(x, y, z,  table, depth)) |> 
    mutate(unscaled_price = prediction * original_sd + original_mean) |> 
    mutate(difference = round(abs(unscaled_price - price))) |> 
    pull(difference) |> quantile(seq(.05, 1, .05))

# This time I'm going to reuse the variables UNSCALED_PRICE, PERCENT_ERROR, 
# PERCENT and DIFFERENCE. I'm going to apply the quantile() to the variable
# DIFFERENCE to get the different quartiles of the variable, and  I'm going to
# group by the variable PERCENT and the code will return multiple outputs for
# each observation. And I'll need to apply data-wrangling to the data, so the
# data format can be readable. And I'll need to apply distinct functions such 
# pivot_wider(), unnest() and pivot_longer() to achieve it
df |> select(-c(x, y, z,  table, depth)) |> 
    mutate(unscaled_price = prediction * original_sd + original_mean) |> 
    mutate(percent_error = (price - unscaled_price) / price * 100) |> 
    mutate(percent = as.integer(abs(percent_error))) |>
    mutate(difference = round(abs(unscaled_price - price))) |> group_by(percent) |> 
    reframe(quant = quantile(difference)) |> pivot_wider(names_from = percent, 
    names_prefix = 'percent_', values_from = quant, values_fn = list) |> 
    unnest(cols = c(percent_0, percent_1, percent_2, percent_3, percent_4, 
    percent_5, percent_6, percent_7, percent_8, percent_9, percent_10, percent_11, 
    percent_12, percent_13, percent_14, percent_15, percent_16, percent_17, 
    percent_18, percent_19, percent_20, percent_21, percent_22, percent_23, 
    percent_24, percent_25, percent_26, percent_27, percent_28, percent_29, 
    percent_30, percent_31, percent_32, percent_34, percent_35, percent_36, 
    percent_37, percent_38, percent_39, percent_40, percent_41, percent_42, 
    percent_43, percent_45, percent_48, percent_53, percent_54, percent_61, 
    percent_63)) |> 
    add_column('quartile' = c('0%', '25%', '50%', '75%', '100%'), .before = 1) |> 
    pivot_longer(cols = -quartile, names_to = 'percent' , values_to = 'values') |> 
    pivot_wider(names_from = quartile, names_prefix = 'quartile_', 
    values_from = values) |> print.data.frame()

