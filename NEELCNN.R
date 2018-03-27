# Reset the workplace
rm(list=ls())
setwd("~/Documents/McMaster/CAS764/Project/")
library(keras)
library(caret)

# Data Preparation from pre-processed
x_train <- readRDS("NEEL2016/NEELx_train.rds")
x_dev <- readRDS("NEEL2016/NEELx_dev.rds")
x_test <- readRDS("NEEL2016/NEELx_test.rds")
y_train <- readRDS("NEEL2016/NEELy_train.rds")
y_dev <- readRDS("NEEL2016/NEELy_dev.rds")
y_test <- readRDS("NEEL2016/NEELy_test.rds")
y_truth <-readRDS("NEEL2016/NEELy_Truth.rds")

# Below CNN model is from the Keras MNIST CNN example

set.seed(116)  # for reproductibility

# Input image dimensions: rows = feature items, cols = rev+nxt+1

img_rows <- 10 
img_cols <- 10
input_shape <- c(img_rows, img_cols, 1)

# Parameters

batch_size <- 128
num_classes <- 8     # 7 levels of entities 1:7 but python starts from 0
epochs <- 20

# Define model
model <- keras_model_sequential() %>%
        layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                      input_shape = input_shape) %>% 
        layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
        layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
        layer_dropout(rate = 0.25) %>% 
        layer_flatten() %>% 
        layer_dense(units = 128, activation = 'relu') %>% 
        layer_dropout(rate = 0.5) %>% 
        layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
        loss = loss_categorical_crossentropy,
        optimizer = optimizer_adadelta(),
        metrics = c('accuracy')
)

# Train model and measure running time

start_time <- Sys.time()
model %>% fit(
        x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = list(x_dev, y_dev)
)
end_time <- Sys.time()

scores <- model %>% evaluate(x_test, y_test, verbose = 0)

# Output metrics
start_time
end_time

# Report of three test groups
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')

# Further reports - report precision, recall and F1 score
y_pred <- model %>% predict_classes(x_test) %>% factor(levels = c(1:7))

entitylevel <- c("Character","Event","Location", 
                 "Organization", "Person","Product","Thing"  )
levels(y_pred) <- entitylevel

y_truth <- readRDS('NEEL2016/NEELy_truth.rds')

mx <- confusionMatrix(y_pred, y_truth)

cat('Test details: \n')
mx
mx[4]
