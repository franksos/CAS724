# This prep is to consolidate NER categories (to LOC, MISC, ORG and PER)
# and this data will be used to peform bootstrapping and ensemble in ML algorithms

setwd("~/Documents/McMaster/CAS764/Project/")

sink("Reports/BOOT raw R result.txt")

library(dplyr)
library(stringr)
library(keras)
library(caret)
library(modeest) # to find the most common value in bootstrap

#load the data
dev_g <- read.csv("NEEL2016/NEEL2016-dev_neel.gs", sep = "\t", 
                  header = F, stringsAsFactors = F, 
                  col.names = c("id","start","end","url","conf","entity"))
dev_t <- read.csv("NEEL2016/NEEL2016-dev.tsv", sep = "|",
                  header = F, stringsAsFactors = F, 
                  col.names = c("v1","id","v3","twt","v5"))
train_g <- read.csv("NEEL2016/NEEL2016-training_neel.gs", sep = "\t", 
                  header = F, stringsAsFactors = F, 
                  col.names = c("id","start","end","url","conf","entity"))
train_t <- read.csv("NEEL2016/NEEL2016-training.tsv", sep = "|",
                  header = F, stringsAsFactors = F, 
                  col.names = c("v1","id","v3","twt","v5"))
test_g <- read.csv("NEEL2016/NEEL2016-test_neel.gs", sep = "\t", 
                  header = F, stringsAsFactors = F, 
                  col.names = c("id","start","end","url","conf","entity"))
test_t <- read.csv("NEEL2016/NEEL2016-test.tsv", sep = "|",
                  header = F, stringsAsFactors = F, 
                  col.names = c("v1","id","v3","twt","v5"))

#Seems one entity comes with an following ID. 
#Have a look at the record (found it is line 4806). 
#Look at the next record (4807) it seems the next line id is still intact. 
#Then clean up the entity:

train_g[train_g$entity == "Organization373937812812615000",]
train_g[4806,]
train_g[4807,]
train_g[train_g$entity == "Organization373937812812615000",]$entity <- "Organization"

# Now all the dataset has 7 classes of entities. This will be the y_axis classifiers.

# merge dataset and locate the target word

dev <- merge(dev_g, dev_t, by = "id")
train <- merge(train_g, train_t, by = "id")
test <- merge(test_g, test_t, by = "id")

dev <- mutate(dev, ds = "dev")
train <- mutate(train, ds = "train")
test <- mutate(test, ds = "test")

ds <- rbind(train, dev, test)

# Notice we loss some rows of records in training dataset and testing dataset 
# (id don't match in _g and _t dataset)

# define 4 extra features of the data. 
ds$word <- substr(ds$twt, ds$start, ds$end)
ds$cab <- grepl("@",ds$word)
ds$pound <- grepl("#",ds$word)
ds$cap <- grepl("[:upper:]", ds$word)
# after extraction, clean up the word to contain lower letters only
ds$word <- str_replace_all(ds$word, "[^[:alpha:]]", "")
ds$word <- str_to_lower(ds$word)
# find the context before and after the word
ds$before <- substr(ds$twt, 1, ds$start - 1)
ds$after <- substr(ds$twt, ds$end + 1, nchar(ds$twt))
# clean up before and after strings to contain lower letters only
ds$before <- str_replace_all(ds$before, "[^[:alpha:]]", "")
ds$before <- str_to_lower(ds$before)
ds$after <- str_replace_all(ds$after, "[^[:alpha:]]", "")
ds$after <- str_to_lower(ds$after)
# choose 40 characters from each of the clean "before" and "after" sequence
ds$before <- str_sub(ds_before, start = -40)
ds$after <- str_sub(ds_after, end = 40)

# split word into letters, choose which piece to take
# with option to include or exclude the number of characters
split_to_letter <- function (input, st_pos, end_pos, inc_nchar = FALSE) {
        x <- as.character(input)
        n_chr <- nchar(x)
        # split word into letters
        xmat<-strsplit(x, split = NULL)
        maxlen <- max(sapply(xmat,length))
        fillchar <- NA
        xe <- do.call(rbind,lapply(xmat, function(x) c(x, rep(fillchar, maxlen - length(x)))))
        # optimize for performance: discard strings longer than 16 letters
        xe <- xe[,st_pos:end_pos]
        xe2 <- xe; mode(xe2) <- "numeric"; xe2[] <- match(xe, letters)
        # change letter to numbers, fill NA with 0, the matrix cell has value of 0-26
        # Normalize value to be between 0-1
        xe2 <- xe2/26
        xe2[is.na(xe2)] <- 0
        # Normalize nchar - # of characters feature in the word
        nc <- n_chr / maxlen
        nc[is.na(nc)] <- 0
        if (inc_nchar == FALSE) {
                return(xe2)
        }
        if (inc_nchar == TRUE) {
                return(list(xe2, nc))
        }
}

# Split word into letters, numerize and standardize, with nchar info
word_array <- split_to_letter(ds$word, 1, 16, inc_nchar = TRUE)
# Normalize @, # and Uppercase T/F features (1=T 0=F)
cab <- as.integer(ds$cab)
pound <- as.integer(ds$pound)
cap <- as.integer(ds$cap)

# the 40 characters before the word
before_array <- split_to_letter(ds$before,1,40)
after_array <- split_to_letter(ds$after,1,40)

# name classification of entities; Combine the 7 categories to 4
ds$entity[ds$entity == "Person" | ds$entity == "Character"] <- "PER"
ds$entity[ds$entity == "Organization"] <- "ORG"
ds$entity[ds$entity == "Product" | ds$entity == "Thing" | ds$entity == "Event"] <- "MISC"
ds$entity[ds$entity == "Location"] <- "LOC"

entitylevel <- c("LOC","MISC","ORG","PER")
ds$entity <- factor(ds$entity, levels = entitylevel)

# combine features
feature <- cbind.data.frame(word_array[[1]], word_array[[2]], 
                            cab, pound, cap, before_array, after_array, 
                            ds$entity, ds$ds)
colnames(feature) <- c(1:100, "entity", "ds")

saveRDS(feature, file = "Boot/NEELFeature_Consolidate.rds")

rm(list=ls())

feature <- readRDS("Boot/NEELFeature_Consolidate.rds")

# With bootstrapping strategy, combine train and dev data, and use 20% for validation
fea_train <- filter(feature, `ds` != "test")
fea_test <- filter(feature, `ds` == "test")

# Set test dataset
x_test <- as.matrix(select(fea_test, 1:100))
dim(x_test) <- c(dim(x_test)[1],10,10,1)
y_test <- to_categorical(fea_test$entity)

# Set original training set as bs_0. the x_bs_0 and y_bs_0 are processed training data
# Bootstrapped training set as bs_1 to bs_20 accordingly.
bs_0 <- fea_train

x_bs_0 <- as.matrix(select(bs_0, 1:100))
dim(x_bs_0) <- c(dim(x_bs_0)[1],10,10,1)
y_bs_0 <- to_categorical(bs_0$entity)

# Bootstrap 20 samples with replacement
set.seed(116)
bs_1 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_2 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_3 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_4 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_5 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_6 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_7 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_8 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_9 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_10 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_11 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_12 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_13 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_14 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_15 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_16 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_17 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_18 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_19 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]
bs_20 <- fea_train[sample(nrow(fea_train), 6771, replace = T), ]

#set x axis and y axis for the 20 bootstrap datasets
x_bs_1 <- as.matrix(select(bs_1, 1:100))
dim(x_bs_1) <- c(dim(x_bs_1)[1],10,10,1)
y_bs_1 <- to_categorical(bs_1$entity)

x_bs_2 <- as.matrix(select(bs_2, 1:100))
dim(x_bs_2) <- c(dim(x_bs_2)[1],10,10,1)
y_bs_2 <- to_categorical(bs_2$entity)

x_bs_3 <- as.matrix(select(bs_3, 1:100))
dim(x_bs_3) <- c(dim(x_bs_3)[1],10,10,1)
y_bs_3 <- to_categorical(bs_3$entity)

x_bs_4 <- as.matrix(select(bs_4, 1:100))
dim(x_bs_4) <- c(dim(x_bs_4)[1],10,10,1)
y_bs_4 <- to_categorical(bs_4$entity)

x_bs_5 <- as.matrix(select(bs_5, 1:100))
dim(x_bs_5) <- c(dim(x_bs_5)[1],10,10,1)
y_bs_5 <- to_categorical(bs_5$entity)

x_bs_6 <- as.matrix(select(bs_6, 1:100))
dim(x_bs_6) <- c(dim(x_bs_6)[1],10,10,1)
y_bs_6 <- to_categorical(bs_6$entity)

x_bs_7 <- as.matrix(select(bs_7, 1:100))
dim(x_bs_7) <- c(dim(x_bs_7)[1],10,10,1)
y_bs_7 <- to_categorical(bs_7$entity)

x_bs_8 <- as.matrix(select(bs_8, 1:100))
dim(x_bs_8) <- c(dim(x_bs_8)[1],10,10,1)
y_bs_8 <- to_categorical(bs_8$entity)

x_bs_9 <- as.matrix(select(bs_9, 1:100))
dim(x_bs_9) <- c(dim(x_bs_9)[1],10,10,1)
y_bs_9 <- to_categorical(bs_9$entity)

x_bs_10 <- as.matrix(select(bs_10, 1:100))
dim(x_bs_10) <- c(dim(x_bs_10)[1],10,10,1)
y_bs_10 <- to_categorical(bs_10$entity)

x_bs_11 <- as.matrix(select(bs_11, 1:100))
dim(x_bs_11) <- c(dim(x_bs_11)[1],10,10,1)
y_bs_11 <- to_categorical(bs_11$entity)

x_bs_12 <- as.matrix(select(bs_12, 1:100))
dim(x_bs_12) <- c(dim(x_bs_12)[1],10,10,1)
y_bs_12 <- to_categorical(bs_12$entity)

x_bs_13 <- as.matrix(select(bs_13, 1:100))
dim(x_bs_13) <- c(dim(x_bs_13)[1],10,10,1)
y_bs_13 <- to_categorical(bs_13$entity)

x_bs_14 <- as.matrix(select(bs_14, 1:100))
dim(x_bs_14) <- c(dim(x_bs_14)[1],10,10,1)
y_bs_14 <- to_categorical(bs_14$entity)

x_bs_15 <- as.matrix(select(bs_15, 1:100))
dim(x_bs_15) <- c(dim(x_bs_15)[1],10,10,1)
y_bs_15 <- to_categorical(bs_15$entity)

x_bs_16 <- as.matrix(select(bs_16, 1:100))
dim(x_bs_16) <- c(dim(x_bs_16)[1],10,10,1)
y_bs_16 <- to_categorical(bs_16$entity)

x_bs_17 <- as.matrix(select(bs_17, 1:100))
dim(x_bs_17) <- c(dim(x_bs_17)[1],10,10,1)
y_bs_17 <- to_categorical(bs_17$entity)

x_bs_18 <- as.matrix(select(bs_18, 1:100))
dim(x_bs_18) <- c(dim(x_bs_18)[1],10,10,1)
y_bs_18 <- to_categorical(bs_18$entity)

x_bs_19 <- as.matrix(select(bs_19, 1:100))
dim(x_bs_19) <- c(dim(x_bs_19)[1],10,10,1)
y_bs_19 <- to_categorical(bs_19$entity)

x_bs_20 <- as.matrix(select(bs_20, 1:100))
dim(x_bs_20) <- c(dim(x_bs_20)[1],10,10,1)
y_bs_20 <- to_categorical(bs_20$entity)

# Define bootcamp CNN model

boot_cnn <- function (x_train_cnn, y_train_cnn, x_test_cnn, y_test_cnn, note) {
        # Below CNN model is from the Keras MNIST CNN example
        set.seed(116)  # for reproductibility
        # Input image dimensions: rows = feature items, cols = rev+nxt+1
        img_rows <- 10 
        img_cols <- 10
        input_shape <- c(img_rows, img_cols, 1)
        # Parameters
        batch_size <- 128
        num_classes <- 5     # 4 levels of entities, but python starts from 0
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
        
        # Train model
        model %>% fit(
                x_train_cnn, y_train_cnn,
                batch_size = batch_size,
                epochs = epochs,
                validation_split = 0.2
        )
        scores <- model %>% evaluate(x_test_cnn, y_test_cnn, verbose = 0)
        # Output metrics
        cat(note, '\n')
        cat('Test loss:', scores[[1]], '\n')
        cat('Test accuracy:', scores[[2]], '\n')
        # Further reports - report precision, recall and F1 score
        y_pred_cnn <- model %>% predict_classes(x_test_cnn) %>% factor(levels = c(1:4))
        entitylevel <- c("LOC","MISC","ORG","PER")
        levels(y_pred_cnn) <- entitylevel
        return(y_pred_cnn)
}

y_pred_0 <- boot_cnn(x_bs_0, y_bs_0, x_test, y_test, "Baseline")

mx <- confusionMatrix(y_pred_0, fea_test$entity)
cat('Test details: baseline \n')
print(mx)
print(mx[4])

y_pred_1 <- boot_cnn(x_bs_1, y_bs_1, x_test, y_test, "boot1")
y_pred_2 <- boot_cnn(x_bs_2, y_bs_2, x_test, y_test, "boot2")
y_pred_3 <- boot_cnn(x_bs_3, y_bs_3, x_test, y_test, "boot3")
y_pred_4 <- boot_cnn(x_bs_4, y_bs_4, x_test, y_test, "boot4")
y_pred_5 <- boot_cnn(x_bs_5, y_bs_5, x_test, y_test, "boot5")
y_pred_6 <- boot_cnn(x_bs_6, y_bs_6, x_test, y_test, "boot6")
y_pred_7 <- boot_cnn(x_bs_7, y_bs_7, x_test, y_test, "boot7")
y_pred_8 <- boot_cnn(x_bs_8, y_bs_8, x_test, y_test, "boot8")
y_pred_9 <- boot_cnn(x_bs_9, y_bs_9, x_test, y_test, "boot9")
y_pred_10 <- boot_cnn(x_bs_10, y_bs_10, x_test, y_test, "boot10")
y_pred_11 <- boot_cnn(x_bs_11, y_bs_11, x_test, y_test, "boot11")
y_pred_12 <- boot_cnn(x_bs_12, y_bs_12, x_test, y_test, "boot12")
y_pred_13 <- boot_cnn(x_bs_13, y_bs_13, x_test, y_test, "boot13")
y_pred_14 <- boot_cnn(x_bs_14, y_bs_14, x_test, y_test, "boot14")
y_pred_15 <- boot_cnn(x_bs_15, y_bs_15, x_test, y_test, "boot15")
y_pred_16 <- boot_cnn(x_bs_16, y_bs_16, x_test, y_test, "boot16")
y_pred_17 <- boot_cnn(x_bs_17, y_bs_17, x_test, y_test, "boot17")
y_pred_18 <- boot_cnn(x_bs_18, y_bs_18, x_test, y_test, "boot18")
y_pred_19 <- boot_cnn(x_bs_19, y_bs_19, x_test, y_test, "boot19")
y_pred_20 <- boot_cnn(x_bs_20, y_bs_20, x_test, y_test, "boot20")

y_pred_mx <- cbind(y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5, y_pred_6, 
                   y_pred_7, y_pred_8, y_pred_9, y_pred_10, y_pred_11, y_pred_12, 
                   y_pred_13, y_pred_14, y_pred_15, y_pred_16, y_pred_17, 
                   y_pred_18, y_pred_19, y_pred_20)
y_boot <- NULL
for (i in 1:nrow(y_pred_mx)) {
        y_boot[i] <- mfv(y_pred_mx[i,]) #use the most common prediction in bootstrap
}
y_boot_cnn <- factor(y_boot, levels = c(1:4))
entitylevel <- c("LOC","MISC","ORG","PER")
levels(y_boot_cnn) <- entitylevel

mx <- confusionMatrix(y_boot_cnn, fea_test$entity)
cat('Test details: bootstrap_cnn (20) \n')
print(mx)
print(mx[4])

# Define bootcamp RNN model

# a customized function to change probilities to classes
prob_to_class <- function (p) {
        k <- 0
        for(i in 1:dim(p)[1]) {
                k[i] <- which(p[i,] == max(p[i,]))
        }
        return (k - 1) # Keras classfication starts from 0 and R from 1
}

boot_rnn <- function (x_train_rnn, y_train_rnn, x_test_rnn, y_test_rnn, note) {
        # Below model is from MNIST RNN example
        set.seed(116)  # for reproductibility
        # Training parameters.
        batch_size <- 32
        num_classes <- 5 # 4 levels of entities, but python starts from 0
        epochs <- 10
        # Embedding dimensions.
        row_hidden <- 128
        col_hidden <- 128
        dim_x_train <- dim(x_train_rnn)
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
        model %>% fit(
                x_train_rnn, y_train_rnn,
                batch_size = batch_size,
                epochs = epochs,
                verbose = 1,
                validation_split = 0.2
        )
        
        # Evaluation
        scores <- model %>% evaluate(x_test_rnn, y_test_rnn, verbose = 0)
        
        # Output metrics
        cat(note, '\n')
        cat('Test loss:', scores[[1]], '\n')
        cat('Test accuracy:', scores[[2]], '\n')
        
        # Further reports - report precision, recall and F1 score
        y_prob_rnn <- model %>% predict(x_test_rnn) 
        y_pred_rnn <- prob_to_class(y_prob_rnn)
        entitylevel <- c("LOC","MISC","ORG","PER")
        y_pred_rnn <- factor(y_pred_rnn, levels = c(1:4))
        levels(y_pred_rnn) <- entitylevel
        return(y_pred_rnn)
        
}

y_pred_0 <- boot_rnn(x_bs_0, y_bs_0, x_test, y_test, "Baseline")
mx <- confusionMatrix(y_pred_0, fea_test$entity)
cat('Test details: baseline \n')
mx
mx[4]

y_pred_1 <- boot_rnn(x_bs_1, y_bs_1, x_test, y_test, "boot1")
y_pred_2 <- boot_rnn(x_bs_2, y_bs_2, x_test, y_test, "boot2")
y_pred_3 <- boot_rnn(x_bs_3, y_bs_3, x_test, y_test, "boot3")
y_pred_4 <- boot_rnn(x_bs_4, y_bs_4, x_test, y_test, "boot4")
y_pred_5 <- boot_rnn(x_bs_5, y_bs_5, x_test, y_test, "boot5")
y_pred_6 <- boot_rnn(x_bs_6, y_bs_6, x_test, y_test, "boot6")
y_pred_7 <- boot_rnn(x_bs_7, y_bs_7, x_test, y_test, "boot7")
y_pred_8 <- boot_rnn(x_bs_8, y_bs_8, x_test, y_test, "boot8")
y_pred_9 <- boot_rnn(x_bs_9, y_bs_9, x_test, y_test, "boot9")
y_pred_10 <- boot_rnn(x_bs_10, y_bs_10, x_test, y_test, "boot10")
y_pred_11 <- boot_rnn(x_bs_11, y_bs_11, x_test, y_test, "boot11")
y_pred_12 <- boot_rnn(x_bs_12, y_bs_12, x_test, y_test, "boot12")
y_pred_13 <- boot_rnn(x_bs_13, y_bs_13, x_test, y_test, "boot13")
y_pred_14 <- boot_rnn(x_bs_14, y_bs_14, x_test, y_test, "boot14")
y_pred_15 <- boot_rnn(x_bs_15, y_bs_15, x_test, y_test, "boot15")
y_pred_16 <- boot_rnn(x_bs_16, y_bs_16, x_test, y_test, "boot16")
y_pred_17 <- boot_rnn(x_bs_17, y_bs_17, x_test, y_test, "boot17")
y_pred_18 <- boot_rnn(x_bs_18, y_bs_18, x_test, y_test, "boot18")
y_pred_19 <- boot_rnn(x_bs_19, y_bs_19, x_test, y_test, "boot19")
y_pred_20 <- boot_rnn(x_bs_20, y_bs_20, x_test, y_test, "boot20")

y_pred_mx <- cbind(y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5, y_pred_6, 
                   y_pred_7, y_pred_8, y_pred_9, y_pred_10, y_pred_11, y_pred_12, 
                   y_pred_13, y_pred_14, y_pred_15, y_pred_16, y_pred_17, 
                   y_pred_18, y_pred_19, y_pred_20)
y_boot <- NULL
for (i in 1:nrow(y_pred_mx)) {
        y_boot[i] <- mfv(y_pred_mx[i,]) #use the most common prediction in bootstrap
}
y_boot_rnn <- factor(y_boot, levels = c(1:4))
entitylevel <- c("LOC","MISC","ORG","PER")
levels(y_boot_rnn) <- entitylevel

mx <- confusionMatrix(y_boot_rnn, fea_test$entity)
cat('Test details: bootstrap_rnn (20) \n')
print(mx)
print(mx[4])
sink()