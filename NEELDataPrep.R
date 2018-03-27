setwd("~/Documents/McMaster/CAS764/Project/")
library(dplyr)
library(stringr)
library(keras)

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

# name classification of entities
entitylevel <- c("Character","Event","Location", 
                 "Organization", "Person","Product","Thing"  )
ds$entity <- factor(ds$entity, levels = entitylevel)

# combine features
feature <- cbind.data.frame(word_array[[1]], word_array[[2]], 
                            cab, pound, cap, before_array, after_array, 
                            ds$entity, ds$ds)
colnames(feature) <- c(1:100, "entity", "ds")

saveRDS(feature, file = "NEEL2016/NEELFeature.rds")

rm(list=ls())

feature <- readRDS("NEEL2016/NEELFeature.rds")

fea_train <- filter(feature, `ds` == "train")
fea_dev <- filter(feature, `ds` == "dev")
fea_test <- filter(feature, `ds` == "test")

x_train <- as.matrix(select(fea_train, 1:100))
x_dev <- as.matrix(select(fea_dev, 1:100))
x_test <- as.matrix(select(fea_test, 1:100))

# change array dimension to behave like 2d pictures to feed in ML algorithms
dim(x_train) <- c(dim(x_train)[1],10,10,1)
dim(x_dev) <- c(dim(x_dev)[1],10,10,1)
dim(x_test) <- c(dim(x_test)[1],10,10,1)

y_train <- to_categorical(fea_train$entity)
y_dev <- to_categorical(fea_dev$entity)
y_test <- to_categorical(fea_test$entity)

#Save for altorithm
saveRDS(x_train, file = "NEEL2016/NEELx_train.rds")
saveRDS(x_dev, file = "NEEL2016/NEELx_dev.rds")
saveRDS(x_test, file = "NEEL2016/NEELx_test.rds")
saveRDS(y_train, file = "NEEL2016/NEELy_train.rds")
saveRDS(y_dev, file = "NEEL2016/NEELy_dev.rds")
saveRDS(y_test, file = "NEEL2016/NEELy_test.rds")
saveRDS(fea_test$entity, file = "NEEL2016/NEELy_truth.rds")

# Reset the workplace
rm(list=ls())

