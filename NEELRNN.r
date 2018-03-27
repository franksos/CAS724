# Reset the workplace
rm(list=ls())
setwd("~/Documents/McMaster/CAS764/Project/")
library(keras)
library(caret)

# a customized function to change probilities to classes
prob_to_class <- function (p) {
        k <- 0
        for(i in 1:dim(p)[1]) {
                k[i] <- which(p[i,] == max(p[i,]))
        }
        return (k - 1) # Keras classfication starts from 0 and R from 1
}

# Data Preparation from pre-processed
x_train <- readRDS("NEEL2016/NEELx_train.rds")
x_dev <- readRDS("NEEL2016/NEELx_dev.rds")
x_test <- readRDS("NEEL2016/NEELx_test.rds")
y_train <- readRDS("NEEL2016/NEELy_train.rds")
y_dev <- readRDS("NEEL2016/NEELy_dev.rds")
y_test <- readRDS("NEEL2016/NEELy_test.rds")
y_truth <-readRDS("NEEL2016/NEELy_Truth.rds")

# Below model is from MNIST RNN example

set.seed(116)  # for reproductibility

# Training parameters.
batch_size <- 32
num_classes <- 8 # 7 levels of entities 1:7 but python starts from 0
epochs <- 10

# Embedding dimensions.
row_hidden <- 128
col_hidden <- 128

dim_x_train <- dim(x_train)
cat('x_train_shape:', dim_x_train)
cat(nrow(x_train), 'train samples')
cat(nrow(x_test), 'test samples')

# Define input dimensions
row <- dim_x_train[[2]]
col <- dim_x_train[[3]]
pixel <- dim_x_train[[4]]

# Model input (4D)
input <- layer_input(shape = c(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper
encoded_rows <- input %>% time_distributed(layer_lstm(units = row_hidden))

# Encodes columns of encoded rows
encoded_columns <- encoded_rows %>% layer_lstm(units = col_hidden)

# Model output
prediction <- encoded_columns %>%
        layer_dense(units = num_classes, activation = 'softmax')

# Define Model

model <- keras_model(input, prediction)
model %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = c('accuracy')
)

# Training model and measure running time
start_time <- Sys.time()
model %>% fit(
        x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        validation_data = list(x_dev, y_dev)
)
end_time <- Sys.time()

# Evaluation
scores <- model %>% evaluate(x_test, y_test, verbose = 0)

# Output metrics
start_time
end_time

# Report of three test groups
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')

# Further reports - report precision, recall and F1 score
y_prob <- model %>% predict(x_test) 
y_pred <- prob_to_class(y_prob)

entitylevel <- c("Character","Event","Location", 
                 "Organization", "Person","Product","Thing")
y_pred <- factor(y_pred, levels = c(1:7))
levels(y_pred) <- entitylevel

mx <- confusionMatrix(y_pred, y_truth)

cat('Test set A details: \n')
mx
mx[4]
