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
x_train <- readRDS("CoNLL2003/CONLLx_train.rds")
y_train <- readRDS("CoNLL2003/CONLLy_train.rds")
x_testa <- readRDS("CoNLL2003/CONLLx_testa.rds")
y_testa <- readRDS("CoNLL2003/CONLLy_testa.rds")
x_testb <- readRDS("CoNLL2003/CONLLx_testb.rds")
y_testb <- readRDS("CoNLL2003/CONLLy_testb.rds")
x_testc <- readRDS("CoNLL2003/CONLLx_testc.rds")
y_testc <- readRDS("CoNLL2003/CONLLy_testc.rds")

# Below model is from MNIST RNN example

set.seed(116)  # for reproductibility

# Training parameters.
batch_size <- 32
num_classes <- 6 # number of y-axis
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
        validation_split = 0.2
)
end_time <- Sys.time()

# Evaluation
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
y_trutha <- readRDS('CoNLL2003/y_trutha.rds')
y_truthb <- readRDS('CoNLL2003/y_truthb.rds')
y_truthc <- readRDS('CoNLL2003/y_truthc.rds')

y_proba <- model %>% predict(x_testa)
y_probb <- model %>% predict(x_testb)
y_probc <- model %>% predict(x_testc)

y_preda <- prob_to_class(y_proba)
y_predb <- prob_to_class(y_probb)
y_predc <- prob_to_class(y_probc)

entitylevel <- c("O", "LOC","MISC","ORG","PER")

y_preda <- factor(y_preda, levels = c(1:5))
y_predb <- factor(y_predb, levels = c(1:5))
y_predc <- factor(y_predc, levels = c(1:5))
levels(y_preda) <- entitylevel
levels(y_predb) <- entitylevel
levels(y_predc) <- entitylevel

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

