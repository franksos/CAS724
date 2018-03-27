# Reset the workplace
rm(list=ls())
setwd("~/Documents/McMaster/CAS764/Project/")
library(keras)
library(caret)

# Data Preparation from pre-processed
x_train <- readRDS("CoNLL2003/CONLLx_train.rds")
y_train <- readRDS("CoNLL2003/CONLLy_train.rds")
x_testa <- readRDS("CoNLL2003/CONLLx_testa.rds")
y_testa <- readRDS("CoNLL2003/CONLLy_testa.rds")
x_testb <- readRDS("CoNLL2003/CONLLx_testb.rds")
y_testb <- readRDS("CoNLL2003/CONLLy_testb.rds")
x_testc <- readRDS("CoNLL2003/CONLLx_testc.rds")
y_testc <- readRDS("CoNLL2003/CONLLy_testc.rds")

# Below CNN model is from the Keras MNIST CNN example

set.seed(116)  # for reproductibility

# Input image dimensions: rows = feature items, cols = rev+nxt+1

img_rows <- 20 
img_cols <- 15
input_shape <- c(img_rows, img_cols, 1)

# Parameters

batch_size <- 128
num_classes <- 6 # 5 levels of entities 1:5 but python starts from 0
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
        validation_split = 0.2
)
end_time <- Sys.time()

scores_a <- model %>% evaluate(x_testa, y_testa, verbose = 0)
scores_b <- model %>% evaluate(x_testb, y_testb, verbose = 0)
scores_c <- model %>% evaluate(x_testc, y_testc, verbose = 0)

# Output metrics
start_time
end_time

# Report of three test groups
cat('Test Group A\n')
cat('Test loss:', scores_a[[1]], '\n')
cat('Test accuracy:', scores_a[[2]], '\n')
cat('Test Group B\n')
cat('Test loss:', scores_b[[1]], '\n')
cat('Test accuracy:', scores_b[[2]], '\n')
cat('Test Group C\n')
cat('Test loss:', scores_c[[1]], '\n')
cat('Test accuracy:', scores_c[[2]], '\n')

# Further reports - report precision, recall and F1 score
y_preda <- model %>% predict_classes(x_testa) %>% factor(levels = c(1:5))
y_predb <- model %>% predict_classes(x_testb) %>% factor(levels = c(1:5))
y_predc <- model %>% predict_classes(x_testc) %>% factor(levels = c(1:5))

entitylevel <- c("O", "LOC","MISC","ORG","PER")
levels(y_preda) <- entitylevel
levels(y_predb) <- entitylevel
levels(y_predc) <- entitylevel

y_trutha <- readRDS('CoNLL2003/y_trutha.rds')
y_truthb <- readRDS('CoNLL2003/y_truthb.rds')
y_truthc <- readRDS('CoNLL2003/y_truthc.rds')

mx_a <- confusionMatrix(y_preda, y_trutha)
mx_b <- confusionMatrix(y_predb, y_truthb)
mx_c <- confusionMatrix(y_predc, y_truthc)

cat('Test set A details: \n')
mx_a
mx_a[4]
cat('Test set B details: \n')
mx_b
mx_b[4]
cat('Test set C details: \n')
mx_c
mx_c[4]
