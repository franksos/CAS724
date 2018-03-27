setwd("~/Documents/McMaster/CAS764/Project/")
library(dplyr)
library(stringr)
library(keras)

train <- read.csv("CoNLL2003/eng.train", sep = " ", stringsAsFactors = F)
testa <- read.csv("CoNLL2003/eng.testa", sep = " ", stringsAsFactors = F)
testb <- read.csv("CoNLL2003/eng.testb", sep = " ", stringsAsFactors = F)
testc <- read.csv("CoNLL2003/eng.testc", sep = " ", header = F, stringsAsFactors = F)

names(train)<- c("word","POS","chunk","entity")
names(testa)<- c("word","POS","chunk","entity")
names(testb)<- c("word","POS","chunk","entity")
names(testc)<- c("word","POS","chunk","entity")

train <- mutate(train, ds = "train")
testa <- mutate(testa, ds = "testa")
testb <- mutate(testb, ds = "testb")
testc <- mutate(testc, ds = "testc")
ds <- rbind(train, testa, testb, testc)

## only use letters

ds$word <- str_replace_all(ds$word, "[^[:alpha:]]", "")
ds$word <- str_to_lower(ds$word)
ds$POS <- str_replace_all(ds$POS, "[^[:alpha:]]", "")
ds$chunk <- str_replace_all(ds$chunk, "[^[:alpha:]]", "")

ds$nchar <- nchar(ds$word)

unique(ds$entity)
ds$entity[ds$entity == "I-ORG" | ds$entity == "B-ORG"] <- "ORG"
ds$entity[ds$entity == "I-LOC" | ds$entity == "B-LOC"] <- "LOC"
ds$entity[ds$entity == "I-PER" | ds$entity == "B-PER"] <- "PER"
ds$entity[ds$entity == "I-MISC" | ds$entity == "B-MISC"] <- "MISC"
ds$entity[ds$entity == "O" | ds$entity == ""] <- "O"
entitylevel <- c("O", "LOC","MISC","ORG","PER")

## uniqword <- length(unique(ds$word))
## tok <- text_tokenizer(num_words = uniqword) %>% fit_text_tokenizer(ds$word)
## this below command causes error 
## t_x <- texts_to_matrix(tok, ds$word, mode = 'tfidf')
## use word2vec to pre-process also causes error
## therefore need other ways to transform data. Shown below.

# End of phase one: Save data for future use
write.csv(ds, 'CoNLL2003/totaldata.txt')

# Reset the workplace
rm(list=ls())

# Reload data, read all strings as characters
ds <- read.csv("CoNLL2003/totaldata.txt", stringsAsFactors = F)

x <- as.character(ds$word)
# split word into letters
xmat<-strsplit(x, split = NULL)
maxlen <- max(sapply(xmat,length))
fillchar <- NA
xe <- do.call(rbind,lapply(xmat, function(x) c(x, rep(fillchar, maxlen - length(x)))))
# optimize for performance: discard strings longer than 18 letters
xe <- xe[,1:17]
xe2 <- xe; mode(xe2) <- "numeric"; xe2[] <- match(xe, letters)
# change letter to numbers, fill NA with 0, the matrix cell has value of 0-26
# Normalize value to be between 0-1
xe2 <- xe2/26
xe2[is.na(xe2)] <- 0
# Normalize nchar - # of characters
nc <- ds$nchar / maxlen
nc[is.na(nc)] <- 0
# Normalize POS tag
ds$POS <- factor(ds$POS)
pos <- as.integer(ds$POS) / max(as.integer(ds$POS))
pos[is.na(pos)] <- 0
# Normalize chunk tag
ds$chk <- factor(ds$chk)
chk <- as.integer(ds$chunk) / max(as.integer(ds$chunk))
chk[is.na(chk)] <- 0
# combine xe, #of characters, pos and chk (altogether 20 features), and add ds file entity (y-axis)
# as well as train / test subset info
# make entity a factor with global levels
entitylevel <- c("O", "LOC","MISC","ORG","PER")
ds$entity <- factor(ds$entity, levels = entitylevel)
feature <- cbind.data.frame(xe2, nc, pos, chk, ds$entity, ds$ds)

# End of phase two: Save data for future
saveRDS(feature, file = "CoNLL2003/CONLLFeature.rds")
# Reset the workplace
rm(list=ls())


## This function expends a line of numerified word to a matrix (x)
## using (prev) number of previous words and (nxt) number of next words.
## to mimic a greyscale pic (2D pixels, 1D = 1 for greyscale)
## for CNN/RNN algorithm
adjacentmtx <- function(x, prev, nxt ) {
        
        d <- dim(x)
        if(length(d) != 2) stop("input not 2D with tuple and attribute")
        
        xt_prev <- vector("list", prev)
        xt_nxt <- vector("list", nxt)
        xt <- as.vector(t(x))
        
        for (i in 1 : prev) {
                xt_prev[[i]] <- c(rep(0,d[2]*i), xt[1:(length(xt)-d[2]*i)])
                dim(xt_prev[[i]]) <- c(d[2],d[1])
                x <- c(t(xt_prev[[i]]),x)
        }
        
        for (i in 1 : nxt) {
                xt_nxt[[i]] <- c(xt[(d[2]*i + 1):(length(xt))],rep(0, d[2]*i))
                dim(xt_nxt[[i]]) <- c(d[2],d[1])
                x <- c(x, t(xt_nxt[[i]]))
        }
        
        dim(x) <- c(d[1],d[2],(prev+nxt+1),1)
        return(x)
}

feature <- readRDS("CoNLL2003/CONLLFeature.rds")

fea_train <- filter(feature, `ds$ds` == "train")
fea_testa <- filter(feature, `ds$ds` == "testa")
fea_testb <- filter(feature, `ds$ds` == "testb")
fea_testc <- filter(feature, `ds$ds` == "testc")

#TRAINING DATA
train.x <- as.matrix(select(fea_train, 1:20))
x_train <- adjacentmtx(train.x, 7, 7)

train.y <- fea_train[,21]
y_train <- to_categorical(train.y)

#TEST DATA
test.x.a <- as.matrix(select(fea_testa, 1:20))
x_testa <- adjacentmtx(test.x.a, 7, 7)
test.y.a <- fea_testa[,21]
y_testa <- to_categorical(test.y.a)

test.x.b <- as.matrix(select(fea_testb, 1:20))
x_testb <- adjacentmtx(test.x.b, 7, 7)
test.y.b <- fea_testb[,21]
y_testb <- to_categorical(test.y.b)

test.x.c <- as.matrix(select(fea_testc, 1:20))
x_testc <- adjacentmtx(test.x.c, 7, 7)
test.y.c <- fea_testc[,21]
y_testc <- to_categorical(test.y.c)

#Save for altorithm
saveRDS(x_train, file = "CoNLL2003/CONLLx_train.rds")
saveRDS(y_train, file = "CoNLL2003/CONLLy_train.rds")
saveRDS(x_testa, file = "CoNLL2003/CONLLx_testa.rds")
saveRDS(y_testa, file = "CoNLL2003/CONLLy_testa.rds")
saveRDS(x_testb, file = "CoNLL2003/CONLLx_testb.rds")
saveRDS(y_testb, file = "CoNLL2003/CONLLy_testb.rds")
saveRDS(x_testc, file = "CoNLL2003/CONLLx_testc.rds")
saveRDS(y_testc, file = "CoNLL2003/CONLLy_testc.rds")
saveRDS(test.y.a, file = "CoNLL2003/y_trutha.rds")
saveRDS(test.y.b, file = "CoNLL2003/y_truthb.rds")
saveRDS(test.y.c, file = "CoNLL2003/y_truthc.rds")

# Reset the workplace
rm(list=ls())

