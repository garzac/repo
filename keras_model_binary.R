
# The following model is a Binary Classification model that uses the data from
# Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Also I use the libraries
# Tensorflow, Keras and Tidyverse. And the model it will predict, if a case have  
# cancer based in the values of the dataset.

#Load the libraries
library(keras)
library(tidyverse)
library(tensorflow)

# Now reset the  keras  session & set up the random numbers from tensorflow 
tf$keras$backend$clear_session()
tf$keras$utils$set_random_seed(23L)

# I downloaded the dataset from Kaggle. And I read the dataset from the directory
# in my computer and store it in the dataframe data
data <- read_csv('~/data/wdbc_data.csv')

# Verify if the dataframe have duplicate values
sum(duplicated(data))

# The tibble have two variables that are useless, so I eliminate those variables
data <- data |> select(-c(id, ...33))

# I'm going to use the variable diagnosis as the variable-Y.  
# The variable diagnosis is a string variable that have two values: 
# B is for Benign and M is for Malignant. So I'm going to convert it in a binary
# value: 0 is for Benign and 1 is for Malignant. In this step I want to verify
# that both variables have the same number of observations. So I will create a
# temporary variable NEW_DIAGNOSIS to compare the values in both variables
data |> mutate(new_diagnosis = if_else(diagnosis == 'B', 0, 1)) |> 
    count(diagnosis, new_diagnosis)

# Now that I verified that both variables have the same number of observations,
# I'll reuse the previous code to realize the same operation, but this time
# I'm going to to do it over the variable DIAGNOSIS and change the variable as 
# a binary value
data <- data |> mutate(diagnosis = if_else(diagnosis == 'B', 0, 1))

# Now, I'm going to apply normalization on selected variables of the dataset.
# The variable-Y Diagnosis is in the column number one, so I only apply the
# normalization on the rest of the variables, since the column number 2 until  
# the column number 31
data <- data |> mutate(across(c(2:31), scale))

# Now I'm going to create a vector of indices that I'll use to split the 
# dataset data into two subsets of data. But due that the dataset is limited
# (568 rows). I'll just have two subsets of data, one subset is for training and   
# the other subset of data is for testing
set.seed(51)
index <- sample(x = 1:2, size = nrow(data), replace = TRUE, prob = c(.8, .2))
table(index)

# Now, is time to split the dataset into subsets of data.
# Index 1 is for training, Index 2 is for testing
train <- data[index == 1, ]
test <- data[index == 2, ]

# In this step, both datasets will be separated in X and Y. First, I select the 
# variable-Y Diagnosis, then is converted to a matrix and it's stored in train_y.
# Second, I remove the variable-Y Diagnosis, then is converted to a matrix and 
# it's in train_x. This process it is done in both datasets train and test.
train_y <- train |> select(diagnosis) |> as.matrix()
train_x <- train |> select(-diagnosis) |> as.matrix()

test_y <- test |> select(diagnosis) |> as.matrix()
test_x <- test |> select(-diagnosis) |> as.matrix()

# Now, define the keras model with the respective layers, activations, optimizers
# and losses
model <- keras_model_sequential()

model |> layer_dense(units = 24, activation = 'relu', input_shape = ncol(train_x)) |> 
    layer_dense(units = 12, activation = 'relu') |> 
    layer_dense(units = 1, activation = 'sigmoid')

model |> compile(optimizer = optimizer_adam(), loss = loss_binary_crossentropy(),
                metrics = metric_binary_accuracy())

# Now is time for training. When the training is finished I have the next values
# For training the loss is 0.0663 and the accuracy is 0.9867
# For testing the loss is 0.0846 and the accuracy is 0.9915
# I believe that based in the metrics, the model can be used with confidence 
model |> fit(x = train_x, y = train_y, epochs = 20, verbose = 1, 
            validation_data = list(test_x, test_y))

