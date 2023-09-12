# The following code will plot the most common name from babynames dataset by 
# each gender and in the year 2017
library(tidyverse)
library(babynames)
library(gridExtra)

p1 <- babynames |> filter(year == 2017 & sex == 'F') |> slice_max(n, n = 15) |> 
    ggplot(aes(x = n , y = fct_reorder(name, n))) +
    geom_col(fill = "#1F78B4") +
    ggtitle('Most common Female Names in 2017') +
    ylab('Names') +
    xlab('Amount') +
    scale_x_continuous(breaks = seq(2000, 20000, by = 2000), 
    minor_breaks = seq(1000, 19000, by = 2000))

p2 <- babynames |> filter(year == 2017 & sex == 'M') |> slice_max(n, n = 15) |> 
    ggplot(aes(x = n , y = fct_reorder(name, n))) +
    geom_col(fill = "#1F78B4") +
    ggtitle('Most common Male Names in 2017') +
    ylab('Names') +
    xlab('Amount') +
    scale_x_continuous(breaks = seq(2000, 18000, by = 2000), 
    minor_breaks = seq(1000, 19000, by = 2000))

grid.arrange(p1, p2, ncol = 2)

