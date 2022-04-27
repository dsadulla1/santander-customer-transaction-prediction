setwd("D:/Clients/Kaggle/Santander Customer Transaction Prediction/data")

library(data.table)
library(tidyverse)
library(mlr)
library(ggthemr) # install.packages("ggthemr")
library(ggthemes) # install.packages("ggthemes")

df <- fread("./train.csv", stringsAsFactors = FALSE)

head(df)
dim(df)
colnames(df)

table(sapply(df,class))

df_test <- fread("./test.csv", stringsAsFactors = FALSE)

head(df_test)
dim(df_test)
colnames(df_test)

table(sapply(df_test,class))

x <- summarizeColumns(df)


# Predictions

train_results <- fread("../results/TrainResults_Trial1.csv")
colnames(train_results)
dim(train_results)

test_results <- fread("../results/TestResults_Trial1.csv")
colnames(test_results)
dim(test_results)


# distribution of the prediction score grouped by known outcome
# http://ethen8181.github.io/machine-learning/unbalanced/unbalanced.html
ggplot( train_results, aes( pred, color = as.factor(actual) ) ) + 
  geom_density( size = 1 ) +
  ggtitle( "Training Set's Predicted Score" ) + 
  scale_color_economist( name = "data", labels = c( "negative", "positive" ) ) + 
  theme_economist()


# https://developers.google.com/machine-learning/crash-course/classification/thresholding

# Deciling
tapply(train_results$pred, train_results$actual, summary)

n_tile_num = 20

cbind(Count = t(t(tapply(train_results$actual,
                         ntile(desc(train_results$pred),n_tile_num),
                         length))),
      Positives = t(t(tapply(train_results$actual,
                             ntile(desc(train_results$pred),n_tile_num),
                             sum))),
      Avg_Prob = t(t(tapply(train_results$pred,
                            ntile(desc(train_results$pred),n_tile_num),
                            min, na.rm=T))),
      Avg_Prob = t(t(tapply(train_results$pred,
                            ntile(desc(train_results$pred),n_tile_num),
                            mean, na.rm=T))),
      Avg_Prob = t(t(tapply(train_results$pred,
                            ntile(desc(train_results$pred),n_tile_num),
                            max, na.rm=T))))