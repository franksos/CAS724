# This algorithm runs an HMM prediction model for the CONLL dataset
# the HMM.py code credit goes to tripleday:
# https://github.com/tripleday/simple_HMM

# The three test datasets were already individually evaluated in the CNN/RNN algorithm
# and in this HMM algorithm I combine the test files together to evaluate

setwd("~/Documents/McMaster/CAS764/Project/HMM")
sink(file = '../Reports/CONLLHMM raw R result.txt')
starttime <- Sys.time()
system("python HMM.py eng.train eng.testa outputa")
system("python HMM.py eng.train eng.testb outputb")
system("python HMM.py eng.train eng.testc outputc")
endtime <- Sys.time()
setwd("~/Documents/McMaster/CAS764/Project")

library(caret)

outa <- read.csv("HMM/outputa",sep = " ", stringsAsFactors = F)
outb <- read.csv("HMM/outputb",sep = " ", stringsAsFactors = F)
outc <- read.csv("HMM/outputc",sep = " ", stringsAsFactors = F)

#output column 4 is the truth value and column 5 is the predicted value
names(outa)<- c("word","POS","chunk","ent_t", "ent_p")
names(outb)<- c("word","POS","chunk","ent_t", "ent_p")
names(outc)<- c("word","POS","chunk","ent_t", "ent_p")

clean_entity <- function(char) {
        char[char == "I-ORG" | char == "B-ORG"] <- "ORG"
        char[char == "I-LOC" | char == "B-LOC"] <- "LOC"
        char[char == "I-PER" | char == "B-PER"] <- "PER"
        char[char == "I-MISC" | char == "B-MISC"] <- "MISC"
        char[char == "O" | char == ""] <- "O"
        return(char)
}

outa_truth <- clean_entity(outa$ent_t)
outa_pred <- clean_entity(outa$ent_p)
outb_truth <- clean_entity(outb$ent_t)
outb_pred <- clean_entity(outb$ent_p)
outc_truth <- clean_entity(outc$ent_t)
outc_pred <- clean_entity(outc$ent_p)

entitylevel <- c("O", "LOC","MISC","ORG","PER")
outa_truth <- factor(outa_truth, levels = entitylevel)
outa_pred <- factor(outa_pred, levels = entitylevel)
outb_truth <- factor(outb_truth, levels = entitylevel)
outb_pred <- factor(outb_pred, levels = entitylevel)
outc_truth <- factor(outc_truth, levels = entitylevel)
outc_pred <- factor(outc_pred, levels = entitylevel)

print(starttime)
print(endtime)

cat('test a \n')
mx_a <- confusionMatrix(outa_pred, outa_truth)
mx_a
mx_a[4]

cat('test b \n')
mx_b <- confusionMatrix(outb_pred, outb_truth)
mx_b
mx_b[4]

cat('test c \n')
mx_c <- confusionMatrix(outc_pred, outc_truth)
mx_c
mx_c[4]

sink()