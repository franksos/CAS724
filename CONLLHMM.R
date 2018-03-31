# This algorithm runs an HMM prediction model for the CONLL dataset
# the HMM.py code credit goes to https://github.com/manubharghav/NER 

# The three test datasets were already individually evaluated in the CNN/RNN algorithm
# and in this HMM algorithm I combine the test files together to evaluate

setwd("~/Documents/McMaster/CAS764/Project/HMM")
system("cat eng.testa eng.testb eng.testc > eng.test")
system("python HMM.py eng.train eng.test output")
setwd("~/Documents/McMaster/CAS764/Project/")

library(caret)

out <- read.csv("HMM/output",sep = " ", stringsAsFactors = F)

#output column 4 is the truth value and column 5 is the predicted value
names(out)<- c("word","POS","chunk","ent_t", "ent_p")

clean_entity <- function(char) {
        char[char == "I-ORG" | char == "B-ORG"] <- "ORG"
        char[char == "I-LOC" | char == "B-LOC"] <- "LOC"
        char[char == "I-PER" | char == "B-PER"] <- "PER"
        char[char == "I-MISC" | char == "B-MISC"] <- "MISC"
        char[char == "O" | char == ""] <- "O"
        return(char)
}

out_truth <- clean_entity(out$ent_t)
out_pred <- clean_entity(out$ent_p)

entitylevel <- c("O", "LOC","MISC","ORG","PER")
out_truth <- factor(out_truth, levels = entitylevel)
out_pred <- factor(out_pred, levels = entitylevel)

mx <- confusionMatrix(out_pred, out_truth)
mx
mx[4]
