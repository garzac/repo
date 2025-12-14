
# The following model is a Binary Classification model that uses the data from
# Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Also I use the algorithm
# XGBoost. And the model it will predict, if a case have cancer based in the 
# values of the dataset. 

# Loading the required libraries 
library(tidyverse)
library(xgboost)

# I downloaded the dataset from Kaggle. And I read the dataset from the directory
# in my computer and store it in the dataframe data
data <- read_csv('~/data/wdbc_data.csv')

# Verify if the dataframe have duplicate values
sum(duplicated(data))

# The tibble have two variables that are useless, so I eliminate those variables
data <- data |> select(-c(id, ...33))

# I'm going to use the varible diagnosis as the variable-Y.  
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
data <- data |> mutate(diagnosis = if_else(diagnosis == 'B', 0, 1)) |> 
    relocate(diagnosis, .after = last_col())

# Now I'm going to create a vector of indices that I'll use to split the 
# dataset data into two subsets of data. But due that the dataset is limited
# (568 rows). I'll just have two subsets of data, one subset is for training and   
# the other subset of data is for doing predictions
set.seed(51)
index <- sample(x = 1:2, size = nrow(data), replace = TRUE, prob = c(.8, .2))
table(index)

# Now, is time to split the dataset into subsets of data.
# Index 1 is for training, Index 2 is for doing predictions
train <- data[index == 1, ]
to_predict <- data[index == 2, ]

# So the data can work in the XGBoost algorithm, the data it will need to be 
# transformed. First remove the variable-Y (diagnosis) from the dataset. Then  
# convert it to a matrix. And finally do the conversion to a xgb.DMatrix object 
# and in this step, I define the variable-diagnosis as the label. Also convert    
# the dataframe to_predict to a matrix
train <- train |> select(-diagnosis) |> as.matrix() |> 
    xgb.DMatrix(label = train$diagnosis) 

to_predict <- as.matrix(to_predict)

# Is time to train the model. When the training is completed the metrics is: 
# for train-set the loss is: 0.16593 
model <- xgboost(data = train, objective = 'binary:logistic', nrounds = 20, 
        verbose = 1)

 # Now is time to make the predictions with the new data to_predict. When I do
 # the predictions, I eliminate the variable diagnosis that is the column 
 # number 31 and the predictions are stored in the object-prediction
 prediction <- predict(model, to_predict[, -31])

 # Now is time to evaluate the predictions, but first I need to apply data-wrangling.  
 # The object prediction is a vector and the first thing is convert it to a tibble.
 # Then I merge the object prediction with the object to_predict using cbind()
 # and then convert it to a tibble-object using as_tibble(). Then I'm going to 
 # create a new variable CLASS_PREDICTION in which I apply a threshold to classify  
 # if a value of PREDICTION means that have cancer or isn't, the threshold is 0.5
 df <- data.frame(prediction = prediction) |> cbind(to_predict) |> 
        as_tibble() |> mutate(class_prediction = if_else(prediction > .5, 1, 0)) 
 
 # Now it is time for evaluations. The dataset df have 118 rows, also the same
 # dataframe have the variables DIAGNOSIS, PREDICTION and CLASS_PREDICTION. So, 
 # I want to know how accurate were the predictions and for this I use filter() 
 # to compare the values that are EQUAL in the variables DIAGNOSIS and 
 # CLASS_PREDICTION. Also I want  to view the variable PREDICTION and I use the
 # select(). Then use count() to count the number of observations. This
 # code return: 116. So, 116 / 118 = .9830 This means that this model which tries
 # to predict if a person have cancer or isn't, the ACCURACY of the model is 98.3% 
 df |> filter(diagnosis == class_prediction) |> 
     select(prediction, class_prediction, diagnosis) |> count()

 # Now I will calculate the opposite. The operation will be very similar to the
 # previous one, but this time I'll calculate all the values that are UNEQUAL
 # in the variables DIAGNOSIS and CLASS_PREDICTION. Also I want  to view the 
 # variable PREDICTION and I use the select(). Then use count() to count the number 
 # of observations. This code return: 2. So, 2 / 118 = .01694 This means that 
 # this model which tries to predict if a person have cancer or isn't, 
 # the INACCURACY of the model is 1.7%
 df |> filter(diagnosis != class_prediction) |> 
     select(prediction, class_prediction, diagnosis) |> count()
 
 
  