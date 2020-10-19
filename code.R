library(tidyverse)
library(gridExtra) 
data("diamonds")
diamonds$carat_range <- cut(diamonds$carat, 
    breaks = quantile(diamonds$carat), labels = c("I", "II", "III", "IV"))

I <- diamonds %>% filter(carat_range == "I") %>% group_by(cut) %>%
    mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
           max_price = max(price)) %>% ggplot(aes(price, cut)) +
     geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
     geom_point(alpha = .1) +
     geom_point(aes(x = mean_price), color = "red", size = 2) +
     theme_minimal() +
     ggtitle("Carat Range 0.2 ~ 0.4") +
     scale_x_continuous(breaks = c(400, 800, 1200, 1600, 2000, 2400), 
      minor_breaks = c(600, 1000, 1400, 1800, 2200))

 II <- diamonds %>% filter(carat_range == "II") %>% group_by(cut) %>%
    mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
           max_price = max(price)) %>% ggplot(aes(price, cut)) +
    geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
    geom_point(alpha = .1) +
    geom_point(aes(x = mean_price), color = "red", size = 2) +
    theme_minimal() +
    ggtitle("Carat Range 0.4 ~ 0.7") +
    scale_x_continuous(breaks = c(1000, 2000, 3000, 4000, 5000, 6000), 
                       minor_breaks = c(500, 1500, 2500, 3500, 4500, 
                        5500, 6500))
 
  III <- diamonds %>% filter(carat_range == "III") %>% group_by(cut) %>%
     mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
            max_price = max(price)) %>% ggplot(aes(price, cut)) +
     geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
     geom_point(alpha = .1) +
     geom_point(aes(x = mean_price), color = "red", size = 2) +
     theme_minimal() +
     ggtitle("Carat Range 0.7 ~ 1.04") +
     scale_x_continuous(breaks = c(2000, 4000, 6000, 8000, 10000, 12000,
                        14000, 16000, 18000), 
                        minor_breaks = c(1000, 3000, 5000, 7000, 9000, 
                        11000, 13000, 15000, 17000, 19000))
  
   IV <- diamonds %>% filter(carat_range == "IV") %>% group_by(cut) %>%
      mutate(mean_price = mean(price, na.rm = T), min_price = min(price), 
             max_price = max(price)) %>% ggplot(aes(price, cut)) +
      geom_segment(aes(x = min_price, xend = max_price, yend = cut), alpha = .5) +
      geom_point(alpha = .1) +
      geom_point(aes(x = mean_price), color = "red", size = 2) +
      theme_minimal() +
      ggtitle("Carat Range 1.04 ~ 5.01") +
      scale_x_continuous(breaks = c( 2000, 4000, 6000, 8000, 10000,
                        12000, 14000, 16000, 18000), 
                         minor_breaks = c(3000, 5000, 7000, 9000,
                        11000, 13000, 15000, 17000, 19000))
   
   plot <- grid.arrange(I, II, III, IV, ncol = 1)
 
 
