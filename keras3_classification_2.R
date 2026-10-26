
# The following model is using the libraries Keras3 and Tensorflow API, the data
# used is the diamonds dataset. Instead of doing a regression-model, I turn it 
# into a classification-model. Where I break down the variable price and convert 
# it into factors (classes). The model will learn about 15 classes  that are 
# based on the variable price. And the model it will learn the class based on 
# the variables of the diamonds dataset.

# Loading the libraries
library(keras3)
library(tidyverse)
library(tfdatasets, exclude = 'shape')

# Select the diamonds dataset and then remove the duplicates 
data <- distinct(diamonds)

# Now, is time to transform the dataset. First I'm going to relocate the variables
# carat and price as the last variables. Then I'm going to create the variable 
# class by splitting the variable price into intervals of $1200. Ex: Class 1 it'll
# contain values from $326 (min price) to $1200. Class 2 it'll contain values from
# $1201 to $2400, until Class 15 with values from $16801 to $18823 (max value).
# Third I'm going to convert the categorical variables cut, color and clarity
# into character-type. And finally convert the variable class as one-hot-encode
# using the function to_categorical()
data <- data |> relocate(carat, price, .after = last_col()) |>
    mutate(class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800, 6000, 7200, 
        8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000), 
        labels = c(1:15), include.lowest = TRUE)) |> 
     mutate(cut = as.character(cut), color = as.character(color), 
           clarity = as.character(clarity)) |> mutate(class = to_categorical(class))

# Now, I'm going to create a vector of indexes that later I'll be using for 
# making predictions. The size of the dataframe is 2000 rows. 
set.seed(51)
index_to_predict <- sample(x = nrow(data), size = 2000, replace = FALSE)

# Now, I'm going to use the vector indexes to extract the rows from the dataset 
# and store it in the dataset to_predict
to_predict <- data[index_to_predict, ]
# Now, eliminate all the row-numbers in the indexes from the dataset data.
data <- data[-index_to_predict, ]
# Verify that none row of the dataset data exists in the dataset to_predict
intersect(data, to_predict)

# Remove the variable price, because now the label is the variable class
data <- data |> select(-price)

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

# Reset the keras session
clear_session(free_memory = TRUE)

# Now set up the random number for all the environment
set_random_seed(51L)

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
    depth = feature_float_normalized(),
    table = feature_float_normalized(),
    x = feature_float_normalized(),
    y = feature_float_normalized(),
    z = feature_float_normalized(),
    carat = feature_float_normalized()))

feature_space |> adapt(train_data |> select(-class))


# Then I'm going to extract and the create the respective inputs, features and 
# outputs from feature_space to build a new model
inputs <- feature_space$get_inputs()

features <- feature_space$get_encoded_features()

outputs <- features |> layer_dense(units = 128, activation = 'relu') |> 
    layer_dense(units = 64, activation = 'relu') |> 
    layer_dense(units = 15, activation = 'softmax')

# Now build the model using the respectives inputs and outputs
model <-  keras_model(inputs =  inputs, outputs =  outputs)

# Inspect the model's architecture
summary(model)

# Now I compile the model with its optimizer, loss and metric. Because the classes
# of the dataset are unbalanced, for the metrics argument I'll use the metric
# f1_score, that it'll compute metrics globally by counting the total true positives,
# false negatives and false positivies
model |> compile(optimizer = optimizer_adam(), loss = loss_categorical_crossentropy(), 
                 metrics = metric_f1_score(average = 'micro'))

# Next, I'm going to build two distinct callbacks that it'll be used during the
# training. The first callback will monitor the LOSS in validation-set and when 
# LOSS stops decreasing, the callback it will change the learning rate. The other
# callback also will monitor the F1_SCORE in the validation-set and when the 
# F1_SCORE stops increasing the callback it will stop the training and also 
# it'll RESTORE THE BEST WEIGHTS in the training
lr <- callback_reduce_lr_on_plateau(monitor = 'val_loss',
            patience = 40, verbose = 1, mode = 'min', min_lr = .0001)

stopping <- callback_early_stopping(monitor = 'val_f1_score', patience = 60, 
            verbose = 1, mode ='max', restore_best_weights = TRUE)

# Now, it's time for training. I'll use the respective data and the callbacks. 
# Training is for 100 epochs, when the training is finished the callback retrieves
# the weights from epoch 93. The metrics obtained from training are:
# For train dataset: loss = .3723; f1_score = .8531
# For  val  dataset: loss = .4790; f1_score = .8229
history <- model |> fit(x = train_data |>  select(-class), y = train_data$class, 
            epochs = 100, batch_size = 128, 
            validation_data = list(val_data |>  select(-class), val_data$class),
            callbacks = list(lr, stopping), verbose = 2)

# Now inspect in test dataset, if the metrics can be similar:
# For test dataset: loss = .4693; f1_score = .8177
model |> evaluate(test_data |> select(-class), test_data$class)

# Now is time to make the predictions with the dataset to_predict. When I do
# the predictions, I eliminate the variables class and price. And it's stored in 
# the object prediction
prediction <- predict(model, to_predict |> select(-c(class, price)))

# Inspect the object prediction
head(prediction)

# The object prediction is a matrix with size of 15 columns and 2000 rows.
# Every row in the matrix it shows probabilities and if I sum the probabilities
# by each row, it will sum 100. So, I'll need to convert the object prediction into 
# one-hot-encode. To do this, I need to check by each row in which column is the 
# biggest value and convert that  column in ONE and the rest of the columns in 
# ZEROS. To know in which column is the biggest value I use the apply() function, 
# then the vector created it's used to create the one-hot-encode using the diag() 
# function.
max_index <- apply(prediction, 1, which.max)
one_hot_pred <- diag(ncol(prediction))[max_index, ]

# Now the object one_hot_pred it looks like one-hot-encode, but the number of 
# columns remains at 15 columns. So, it'll need to apply data wrangling to 
# transform these 15 columns into just 1 variable with one column
new_pred <- one_hot_pred |> as_tibble() |> unite(V1:V15, col = 'prediction', sep = '')

# The dataset to_predict have the variable class, but also it's necessary to apply
# data wrangling to transform these 15 columns into just 1 variable with one column
target <- to_predict |> select(class) |> as.matrix() 
target <- target |> as_tibble() |> unite(class.1:class.15, col = 'target', sep = '')

# I have the new_pred and the target into tibble format with one column, Now
# I can deploy data analysis to know how accurate were the predictions. To do this
# I join both tibbles into one and then use the filter() function to filter when 
# both variables have the SAME VALUE. Then use count() to count the number of 
# observations. The code returns 1630, so 1630 / 2000 = .815 This means that this 
# model that will try to predict between the 15 classes of the price of the 
# diamonds, have an ACCURACY of 81.5%  
cbind(target, new_pred) |> as_tibble() |> filter(target == prediction) |> count()

# Now, I want to calculate the opposite, the procedure is very similar as the 
# above, but this time I'm going to count all the number of observations that
# are NOT EQUAL between both variables. This code return 370. So,
# 370 / 2000 = .185 This means that this model that will try to predict between
# the 15 classes of the price of the diamonds, have an INACCURACY of 18.5% 
cbind(target, new_pred) |> as_tibble() |> filter(target != prediction) |> count()


