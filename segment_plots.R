
# The following code creates a segment-plot with the data of diamonds dataset. 
# I think that this type of plot can be helpful to graph different values by 
# each type of class. In the next plot, I'm going to split up the variable-carat 
# into quantiles. Then I'll plot the values of the variable-price and each point 
# in the plot is for each value of the variable-cut. To avoid the overcrowding 
# in the plot I set  the parameter alpha = 0.1, this what it does is that 
# each point is represented as a 10%, it means that 10-points in the plot  
# represents one-point. And finally each plot will show the average-price 
# for each value of the variable-cut as a red-dot.

# Load the libraries
library(tidyverse)
library(gridExtra) 

# Load the dataset
data("diamonds")

# Set the random-seed. Also break down the variable-carat into quantiles.
set.seed(51)
diamonds$carat_range <- cut(diamonds$carat, 
    breaks = quantile(diamonds$carat), labels = c("I", "II", "III", "IV"), 
    include.lowest = TRUE)

# This first plot is for the first-quantile of the carat-variable. This covers 
# from carat 0.2 to 0.4 and store the plot in I
I <- diamonds %>% filter(carat_range == "I") %>% group_by(cut) %>%
    mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
           max_price = max(price)) %>% ggplot(aes(price, cut)) +
    geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
    geom_point(alpha = .1) +
    geom_point(aes(x = mean_price), color = "red", size = 3) +
    theme_minimal() +
    ggtitle("Carat Range 0.2 ~ 0.4") +
    scale_x_continuous(breaks = c(400, 800, 1200, 1600, 2000, 2400), 
                       minor_breaks = c(600, 1000, 1400, 1800, 2200))

# This second plot is for the second-quantile of the carat-variable. This covers
# from carat 0.4 to 0.7 and store the plot in II
II <- diamonds %>% filter(carat_range == "II") %>% group_by(cut) %>%
    mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
           max_price = max(price)) %>% ggplot(aes(price, cut)) +
    geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
    geom_point(alpha = .1) +
    geom_point(aes(x = mean_price), color = "red", size = 3) +
    theme_minimal() +
    ggtitle("Carat Range 0.4 ~ 0.7") +
    scale_x_continuous(breaks = c(1000, 2000, 3000, 4000, 5000, 6000), 
                       minor_breaks = c(500, 1500, 2500, 3500, 4500, 
                                        5500, 6500))

# This plot is for the third-quantile of the carat-variable. This covers
# from carat 0.7 to 1.04 and store the plot in III
III <- diamonds %>% filter(carat_range == "III") %>% group_by(cut) %>%
    mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
           max_price = max(price)) %>% ggplot(aes(price, cut)) +
    geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
    geom_point(alpha = .1) +
    geom_point(aes(x = mean_price), color = "red", size = 3) +
    theme_minimal() +
    ggtitle("Carat Range 0.7 ~ 1.04") +
    scale_x_continuous(breaks = c(2000, 4000, 6000, 8000, 10000, 12000,
                                  14000, 16000, 18000), 
                       minor_breaks = c(1000, 3000, 5000, 7000, 9000, 
                                        11000, 13000, 15000, 17000, 19000))

# This final plot is for the forth-quantile of the carat-variable. This covers
# from carat 1.04 to 5.01 and store the plot in IV
IV <- diamonds %>% filter(carat_range == "IV") %>% group_by(cut) %>%
    mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
           max_price = max(price)) %>% ggplot(aes(price, cut)) +
    geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
    geom_point(alpha = .1) +
    geom_point(aes(x = mean_price), color = "red", size = 3) +
    theme_minimal() +
    ggtitle("Carat Range 1.04 ~ 5.01") +
    scale_x_continuous(breaks = c( 2000, 4000, 6000, 8000, 10000,
                                   12000, 14000, 16000, 18000), 
                       minor_breaks = c(3000, 5000, 7000, 9000,
                                        11000, 13000, 15000, 17000, 19000))

# Now it's time to join the four plots into one plot
plot <- grid.arrange(I, II, III, IV, ncol = 1)


