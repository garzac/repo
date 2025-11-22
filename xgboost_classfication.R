
# The following model is a Multi-Class classification model. The algorithm used
# is XGBoost and the dataset that feeds the model is  the diamonds dataset.
# The variable price have a big range of values from $326 to $18823.
# So, I decided to break down the variable price and convert it into classes.
# The variable-Y will have 15 classes, which each class will de divided in 
# intervals of $1200. And the model it will predict the class of the diamonds, 
# which are based on the variable price.

# Loading the required libraries 
library(tidyverse)
library(xgboost)

# Removing the duplicated data
data <- distinct(diamonds)

# Now, I want to analyse the variable price and break down into factors (classes).
# For this I will use mutate() to create a new variable called class. And within
# mutate, I'll use cut() that will break down the variable price into factors.
# Each class is an interval of 1200. And it will have 15 classes.
# Class 1 is from minimum value $326 to the value $1200.
# Class 2 is from value $1201 to the value $2400 and so on, until
# Class 15 is from value $16801 to the max value $18823.
# Also can check the minimum, maximum, quantity and distinct prices by each class.
data |> mutate(class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800, 
    6000, 7200, 8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000),
    labels = c(1:15), include.lowest = TRUE)) |> group_by(class) |> 
    summarise(min_price = min(price), max_price = max(price), quantity = n(), 
    distinct_price = n_distinct(price))

# Now that I analysed the variable class, I´m going to create it. And I'll reuse
# the same code as above to create the variable class.
data <- data |> mutate(class = cut(price, breaks = c(0, 1200, 2400, 3600, 4800, 
    6000, 7200, 8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 19000),
    labels = c(1:15), include.lowest = TRUE))

# The data type of the categorical variables cut, color, clarity  and also class  
# are factors. So, I need the convert it to a numeric data type
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

# Now that the variable-class will be the variable-Y, I'm going to eliminate
# the variable price in both datasets
data <- data |> select(-price)
to_predict <- to_predict |> select(-price)

# Now, convert the dataset to_predict to a matrix
to_predict <- to_predict |> as.matrix()

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
# transformed. First remove the variable-Y (class) from the dataset. Then convert 
# it to a matrix. And finally do the conversion to a xgb.DMatrix object and in 
# this step, I define the variable-class as the label. This process need to be 
# done in both datasets
train <- train |> select(-class) |> as.matrix() |> xgb.DMatrix(label = train$class) 
test <- test |> select(-class) |> as.matrix() |> xgb.DMatrix(label = test$class)

# To evaluate both datasets train and test in the training process, the parameter
# WATCHLIST will be used, and for this I'll create a list-object with both datasets
watchlist <- list(train = train, test= test)

# Now I need to define the distinct classes in the variable-Y, so I use 
# n_distinct() plus one and will be stored in num_classes object
num_classes <- n_distinct(data$class) + 1

# Here, I will list the list of parameters that will fit the model. 
# Use list() function to define the parameter num_classes
params <- list('num_class' = num_classes)

# Is time to train the model. I set up the parameters PARAMS and WATCHLIST 
# to can inspect the metrics in both datasets and the object is stored in model 
# When the training is completed the metrics are: 
# for train-set the loss is: 0.228664 and for the test-set the loss is: 0.46111
model <- xgb.train(params = params, data = train, objective = 'multi:softmax', 
    nrounds = 100, watchlist = watchlist, verbose = 1, print_every_n = 10)

# Now is time to make the predictions with the new data to_predict. When I do
# the predictions, I eliminate the variable class that is the column number 10
# and the predictions are stored in the object-prediction
prediction <- predict(model, to_predict[, -10])

# Now is time to evaluate the predictions, but first I need to apply data-wrangling.  
# The object prediction is a vector and the first thing is convert it to a tibble.
# Then I merge the object prediction with the object to_predict using cbind()
# and then convert it to a tibble-object using as_tibble(). Last relocate the 
# variables  class and  prediction at the end of the tibble and then store it in  
# the object df
df <- data.frame(prediction = prediction) |> cbind(to_predict) |> 
    as_tibble() |> relocate(class, prediction, .after = last_col())

# Now it is time for evaluations. The dataset df have 3000 rows, also the same
# dataframe have the variables CLASS and PREDICTION. So, I want to know how
# accurate was the predictions and for this I use filter() to compare the values  
# that are EQUAL in both variables. Then use count() to count the number of 
# observations. This code return: 2440. So, 2440 / 3000 = .813 This means that
# this model that will try to predict between the 15 classes of the price of the
# diamonds, have an ACCURACY of 81% 
df |> filter(class == prediction) |> count()

# Now I will calculate the opposite. The operation will be very similar to the
# previous one, but this time I'll calculate all the values that are UNEQUAL
# in both variables. This code return: 560. So, 560 / 3000 = .186 This means that
# this model that will try to predict between the 15 classes of the price of the
# diamonds, have an INACCURACY of 19% 
df |> filter(class != prediction) |> count()



