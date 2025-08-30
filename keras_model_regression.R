# The following model is a regression-model. And it uses the libraries Keras,
# Tensorflow and also the Tensorflow-Dataset API that process the data and 
# transform it into the required tensors.  The model will use the diamonds
# dataset and the model it will be predict the price of a diamond based on 
# all the variables of the dataset. 

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

# Verify that none row of the dataset-sample_data exists in the dataset-to_predict
intersect(sample_data, to_predict)

# Now I'm going to create another vector of indices that I'll use to split the 
# dataset-sample_data into three subsets of data for train/validation/test
set.seed(51)
index_tosplit <- sample(x = 3, size = nrow(sample_data), replace = TRUE,
                        prob = c(.7, .15, .15))

# Check the length for each index
table(index_tosplit)
# Check the proportions for each index
prop.table(table(index_tosplit))

# The next three steps is for preparing the Y-label
# First, extract the variable price that will be the Y-label
price <- select(sample_data, price)

# Second, split the price variable. I'll use each of the indices over the Y-label.
# Index 1 is for train-set, index 2 for validation-set, index 3 for test-set
train_y <- price[index_tosplit == 1, ]
val_y <- price[index_tosplit == 2, ]
test_y <- price[index_tosplit == 3, ]

# Last, is to apply normalization in the Y-label using layer_normalization()
train_y <- layer_normalization(train_y, mean = 2, variance = 1)
val_y <- layer_normalization(val_y, mean = 2, variance = 1)
test_y <- layer_normalization(test_y, mean = 2, variance = 1)

# The next two steps is for preparing the X-features
# First, eliminate the variable price from the dataset-sample_data
sample_data <- sample_data |> select(-price)

# Second, is time to split the data into subsets of data
# Index 1 is for train-set, index 2 for validation-set, index 3 for test-set
train_x <- sample_data[index_tosplit == 1, ]
val_x <- sample_data[index_tosplit == 2, ]
test_x <- sample_data[index_tosplit == 3, ]

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
# Define the inputs/outputs of the model and store it in inputs/outputs
inputs <- layer_input_from_dataset(sample_data)

outputs <- inputs %>%  layer_dense_features(dense_features(spec)) |> 
    layer_dense(units = 512, activation = 'relu') |> 
    layer_dense(units = 256, activation = 'relu') |> 
    layer_dense(units = 1, activation = 'relu')

# Now define the model with the inputs and outputs
model <- keras_model(inputs = inputs, outputs = outputs)

# Inspect the model's architecture
summary(model)

# Now compile the model with the respective loss and optimizer. For a regression
# model, the default optimizer is Stochastic-gradient-descent and for the loss
# the default is mean-squared-error but I chose to use the mean-absolute-error.
model |> compile(optimizer = optimizer_sgd(), loss = loss_mean_absolute_error())

# During a while, I built several models to improve the predictions.
# I built a model when the variable-Y was normalized using the scale() function
# and for the loss use the mean-squared-error and the model's loss was near to 
# zero in all the datasets. And then use the predictions and apply inverse-scaling  
# to convert the predictions in values that are in the same scale. 
# But the predictions were not so good.
#
# So, I decide to try something else and normalize the variable-Y with the
# function layer_normalization() from Keras. This approach increased the loss 
# to 206.5586 for train-set and for validation-set to 268.4194. 
# I know that aren't the best losses. But with this approach the predictions 
# increased in a range of 13%. 
model |> fit(x = train_x, y = train_y, batch_size = 128, 
        epochs = 550, verbose = 1, shuffle = TRUE, callbacks = 
        callback_reduce_lr_on_plateau(monitor = 'loss', patience = 25, factor = 0.1,
        mode = 'auto', min_lr = .001), validation_data = list(val_x, val_y))

# Now, it's time to evaluate the model in test dataset. The loss is 268.8495
model |> evaluate(test_x, test_y)

# Now its time for predictions. I'll use the dataset to_predict and for the
# predictions I'll eliminate the variable-price. And the predictions will be 
# stored as a new-variable called prediction in the dataset to_predict.
to_predict$prediction <- predict(model, to_predict %>% select(-price))

# It's time to evaluate the predictions. I'll create the variable PERCENT_ERROR
# to evaluate the difference-in-percent between the prediction and the price.
# Also I'll create another variable PERCENT that will convert the PERCENT_ERROR
# into an integer with absolute-value. Then I'll use count() to count the number
# of observations of each value in the variable PERCENT. Then I'll create the
# variable CUM_SUM that will realize cumulative-sum of the number of observations
# Finally I'll create the variable RATIO that will calculate the percentage of 
# each value in the variable PERCENT by dividing CUM_SUM by the total.
# The ratio of each value in the variable PERCENT are:
# From 0% to 3% the ratio is 50%; from 5% to below the ratio is 68%
# From 7% to below the ratio is 78%; from 10% to below the ratio is 85%
# This means that 85% of the predictions in the dataset,  the PERCENT_ERROR
# is less or equal to 10% 
to_predict |> select(-c(x, y , z)) |> 
    mutate(percent_eror = (price - prediction) / price * 100) |> 
    mutate(percent = as.integer(abs(percent_eror))) |> count(percent) |> 
    mutate(cum_sum = cumsum(n)) |> mutate(ratio = round(cum_sum / 150 * 100, 2)) |> 
    print.data.frame()

# I'll create the variable DIFFERENCE that will calculate the difference between
# the prediction and the price. And the value will be rounded with absolute value
# Also I'll apply the function quantile() to the variable DIFFERENCE to calculate
# the quartiles from 5%, 10%, 15% until 100%. And I get the next values:
# The 25% quartile is $20; the median is $55.5; the 75% quartile is $223 
# The 85% quartile is $413.45; this means that the 85% of the values of DIFFERENCE 
# between the prediction and the price are less or equal to $413.45
to_predict |> select(-c(x, y , z, table, depth)) |> 
    mutate(difference = round(abs(price - prediction)) ) |> 
    pull(difference) |> quantile(seq(.05, 1, .05))

# In the next evaluation, I'm going to reuse the variable DIFFERENCE from the 
# previous code. Then I want to group by the variable CUT. Next, I want to apply
# the quartile() to the variable DIFFERENCE to get the different quartiles of
# the variable, but the code will return multiple outputs by each observation.
# And I'll need to apply data-wrangling to the data, so the data format can be 
# readable. So, also apply the function pivot_wider() and unnest() to achieve it
to_predict |> select(-c(x, y , z, table, depth)) |> 
    mutate(difference = round(abs(price - prediction)) ) |> 
    group_by(cut) |> reframe(quartile = quantile(difference)) |> 
    pivot_wider(names_from = cut, values_from = quartile, values_fn = list) |> 
    unnest(cols = c('Fair', 'Good', 'Ideal', 'Premium', 'Very Good')) |> 
    add_column('quartile' = c('0%', '25%', '50%', '75%', '100%'), .before = 1)

# I know that based in the model's loss is not the best model, but also I think 
# that based on the metrics, such as the 85% of the dataset have a PERCENT_ERROR
# less or equal to 10% and also that the 85% of the values of DIFFERENCE between 
# the prediction and the price is less or equal to $413.45 in all the dataset.
# This model still can be useful for people without knowledge in diamonds, so
# the people can get an idea of what a diamond can cost.
#
# Let's try if with a much bigger dataset, I can get similar metrics.
# So, I get a new tibble with 3000 rows with new data from the remnants-dataset
#
# AFTER BEEN ANALYZED THE BIG_PREDICTION-DATASET THE METRICS IN BOTH DATASETS
# ARE VERY SIMILAR. THE 83% OF THE DATASET BIG_PREDICTION HAVE A PERCENT_ERROR
# LESS OR EQUAL TO 10% AND ALSO THAT THE 85% OF THE VALUES OF DIFFERENCE BETWEEN
# THE PREDICTION AND THE PRICE IS LESS OR EQUAL TO $491 IN ALL THE DATASET.
# I STILL BELIEVE THAT THIS MODEL STILL CAN BE USEFUL FOR PEOPLE WITHOUT 
# KNOWLEDGE IN DIAMONDS, SO THE PEOPLE CAN GET AN IDEA OF WHAT A DIAMOND CAN COST
set.seed(51)
big_prediction <- remnants |> sample_n(size = 3000)

# Also apply the predictions for the new dataset
big_prediction$prediction <- predict(model, big_prediction %>% select(-price))

# Same as above, I'm going to reuse: I'll create the variable PERCENT_ERROR
# to evaluate the difference-in-percent between the prediction and the price.
# Also I'll create another variable PERCENT that will convert the PERCENT_ERROR
# into an integer with absolute-value. Then I'll use count() to count the number
# of observations of each value in the variable PERCENT. Then I'll create the
# variable CUM_SUM that will realize cumulative-sum of the number of observations
# Finally I'll create the variable RATIO that will calculate the percentage of 
# each value in the variable PERCENT by dividing CUM_SUM by the total.
# The ratio of each value in the variable PERCENT are:
# From 0% to 3% the ratio is 47%; from 5% to below the ratio is 63%
# From 7% to below the ratio is 73%; from 10% to below the ratio is 83%
# This means that 83% of the predictions in the dataset,  the PERCENT_ERROR
# is less or equal to 10% 
big_prediction |> select(-c(x, y , z, depth, table)) |> 
    mutate(percent_eror = abs(price - prediction) / price * 100) |> 
    mutate(perc = as.integer(percent_eror)) |> count(perc) |> 
    mutate(sums = cumsum(n))|> mutate(ratio = round(sums / 3000 * 100, 2)) |> 
    print.data.frame()

# Same as above, I'm going to reuse:
# I'll create the variable DIFFERENCE that will calculate the difference between
# the prediction and the price. And the value will be rounded with absolute value
# Also I'll apply the function quantile() to the variable DIFFERENCE to calculate
# the quartiles from 5%, 10%, 15% until 100%. And I get the next values:
# The 25% quartile is $29; the median is $94; the 75% quartile is $296 
# The 85% quartile is $491; this means that the 85% of the values of DIFFERENCE 
# between the predictions and the price are less or equal to $491
big_prediction |> select(-c(x, y , z, table, depth)) |> 
    mutate(difference = round(abs(price - prediction)) ) |> 
    pull(difference) |> quantile(seq(.05, 1, .05))

# Same as above, I'm going to reuse the variable DIFFERENCE from the 
# previous code. Then I want to group by the variable CUT. Next, I want to apply
# the quartile() to the variable DIFFERENCE to get the different quartiles of
# the variable, but the code will return multiple outputs by each observation.
# And I'll need to apply data-wrangling to the data, so the data format can be 
# readable. So, also apply the function pivot_wider() and unnest() to achieve it
big_prediction |> select(-c(x, y , z, table, depth)) |> 
    mutate(percent_eror = abs(price - prediction) / price * 100) |> 
    mutate(difference = round(abs(price - prediction)))  |> 
    group_by(cut) |> reframe(quantile = quantile(difference)) |> 
    pivot_wider(names_from = cut, values_from = quantile, values_fn = list) |> 
    unnest(cols = c('Fair', 'Good', 'Ideal', 'Premium', 'Very Good')) |> 
    add_column('quartile' = c('0%', '25%', '50%', '75%', '100%'), .before = 1)

# I'm going to reuse the variables PERCENT_ERROR, PERCENT, DIFFERENCE. The
# operation will be similar as the previous code, I'm going to apply the quantile()
# to the variable DIFFERENCE to get the different quartiles of the variable, but 
# this time I'm going to group by the variable PERCENT, and the code will return 
# multiple outputs by each observation. And I'll need to apply data-wrangling to 
# the data, so the data format can be readable. But this time, I'll need to apply 
# distinct functions such pivot_wider(), unnest() and pivot_longer() to achieve it
big_prediction |> select(-c(x, y , z, table, depth)) |> 
    mutate(percent_eror = (price - prediction) / price * 100) |> 
    mutate(percent = abs(as.integer(percent_eror))) |> 
    mutate(difference = round(abs(price - prediction)) ) |> 
    group_by(percent) |> reframe(quant = quantile(difference)) |> 
    pivot_wider(names_from = percent, names_prefix = 'percent_', values_from = quant, 
    values_fn = list) |> unnest(cols = c(percent_0, percent_1, percent_2, percent_3, 
        percent_4, percent_5,percent_6, percent_7, percent_8, percent_9, percent_10, 
        percent_11, percent_12, percent_13, percent_14, percent_15, percent_16, 
        percent_17, percent_18, percent_19, percent_20, percent_21, percent_22, 
        percent_23, percent_24, percent_25, percent_26, percent_27, percent_28, 
        percent_29, percent_30, percent_31, percent_32, percent_33, percent_34, 
        percent_35, percent_37, percent_38, percent_39, percent_40, percent_41,
        percent_42, percent_43, percent_46, percent_48, percent_49, percent_52, 
        percent_60)) |> 
    add_column('quartile' = c('0%', '25%', '50%', '75%', '100%'), .before = 1) |> 
    pivot_longer(cols = -quartile, names_to = 'percent', values_to = 'values') |> 
    pivot_wider(names_from = quartile, names_prefix = 'quartile_' , 
        values_from = values) |> print.data.frame()
