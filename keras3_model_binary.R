
# The following model is a Binary Classification model that uses the data from
# Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Also I use the libraries
# Tensorflow, Keras3 and Tidyverse. And the model it will predict, if a case have  
# cancer based in the values of the dataset.

# Besides, due to the small size of the dataset, I'm going to use the approach
# K-FOLD VALIDATION, where I'll split the dataset into K-partitions through 
# iterations. With each iteration the train and validation datasets will change 
# every time. Then the model will start the training. Next after each training,
# I will obtain each of the metrics to compare them  between each training step.
# And all this process will repeat four times.

# Loading the libraries
library(keras3)
library(tidyverse)
library(tensorflow, exclude = c("shape", "set_random_seed"))

# Reset the keras session
clear_session(free_memory = TRUE)

# Now set up the random number for all the environment
set_random_seed(51L)

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
data <- data |> mutate(across(c(2:31), ~ scale(.) %>% as.vector))

# Now, I'm going to separate the variables into the Features(X) and the Label(Y)
# Also, it need to convert both dataframes into matrices.
x <- data |> select(-diagnosis) |> as.matrix()
y <- data |> select(diagnosis) |> as.matrix()

# Now, I define the model of the Neural Network as a function. Also, here I 
# specify its respective optimizer, loss and metrics of the model. I choose this
# approach because during training I will use a loop for training.
get_model <- function() { 
    model <- keras_model_sequential(input_shape = ncol(x)) |>
        layer_dense(24, activation = "relu") |>
        layer_dense(12, activation = "relu") |>
        layer_dense(1, activation = 'sigmoid')
    model |> compile(optimizer = optimizer_adam(),
        loss = loss_binary_crossentropy(), metrics = metric_binary_accuracy())
    return(model)
}

# Here I define the hyperparameters of the model. Also in this step also I'll 
# create an index-vector that it'll be used to split the dataset into K-partitions
k <- 4
index <- sample(rep(1:k, length.out = nrow(data)))
num_epochs <- 20
train_losses <- numeric(k)
train_accuracies <- numeric(k)
val_losses <- numeric(k)
val_accuracies <- numeric(k)

# This step include different tasks. First, I define a for-loop that it will
# iterate the entire process four times. Second, using the which() function 
# will extract the row-number of the current index. Ex: If the current index is 1, 
# the which function will indicate each of the row-number where the index is equal 
# to 1. And it's stored in fold_indices. Third, use the fold_indices vector to 
# split the dataset into train/validation, so in every iteration both datasets
# will be totally different. Fourth, I build the model using the previous function
# Fifth, the model starts to training. Sixth, When the training stops, I will 
# retrieve the last index of all the metrics. And each metric it'll be stored in 
# its respective object. And all this process it will execute four times.
for (i in 1:k) {
    cat(sprintf("Processing fold #%i\n", i))
    fold_indices <- which(index == i)
    fold_val_x <- x[fold_indices, ] 
    fold_val_y <- y[fold_indices, ] 
    fold_train_x <- x[-fold_indices, ] 
    fold_train_y <- y[-fold_indices, ] 
    model <- get_model() 
    history <- model |> fit( fold_train_x, fold_train_y, epochs = num_epochs, 
            validation_data = list(fold_val_x, fold_val_y), verbose = 0 )
    train_losses[i] <- history$metrics$loss[20]
    train_accuracies[i] <- history$metrics$binary_accuracy[20]
    val_losses[i] <- history$metrics$val_loss[20]
    val_accuracies[i] <- history$metrics$val_binary_accuracy[20]
}

# Now that training finished and now I have all the metrics. It's time to compare
# the LOSS and ACCURACY in both datasets during each training step. 
# The LOSS in both datasets and in every training step are very close to zero.
# The ACCURACY in both datasets and in every training step are above 96%  
str_glue('The  loss  in  train  dataset  during the training step number {1:4} is: {round(train_losses, 6)}', .sep = '\n')

str_glue('The loss in validation dataset during the training step number {1:4} is: {round(val_losses, 6)}', .sep = '\n')

str_glue('The  accuracy  in  train  dataset  during the training step number {1:4} is: {round(train_accuracies, 6)}', .sep = '\n')

str_glue('The accuracy in validation dataset during the training step number {1:4} is: {round(val_accuracies, 6)}', .sep = '\n')

