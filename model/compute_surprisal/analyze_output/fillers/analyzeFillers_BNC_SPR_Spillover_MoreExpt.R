library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lme4)
library(dplyr)
model = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3_Fillers.py.tsv", sep="\t") %>% mutate(wordInItem = Region+1) %>% mutate(item = paste("Filler", Sentence, sep="_")) %>% group_by(item, wordInItem, Region, Word, Script, ID, predictability_weight, deletion_rate) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE))

print(summary(model))


model$LowerCaseToken = tolower(model$Word)
model$WordLength = nchar(as.character(model$Word))



word_freq_50000 = read.csv("stimuli-coca-frequencies.tsv", sep="\t", quote=NULL)
word_freq_50000$LogWordFreq_COCA = log(word_freq_50000$Frequency)


model = merge(model, word_freq_50000, by=c("LowerCaseToken"), all=TRUE)


word_freq_50000 = read.csv("stimuli-bnc-frequencies.tsv", sep="\t", quote=NULL)
word_freq_50000$LogWordFreq = log(word_freq_50000$Frequency)

model = merge(model, word_freq_50000, by=c("LowerCaseToken"), all=TRUE)
#crash()

model$LogWordFreq_COCA.R = resid(lm(LogWordFreq_COCA~LogWordFreq, data=model, na.action=na.exclude))

# Read RT data
fillers_names = read.csv("fillers-names-VSLK.tsv", sep="\t")
                                                                                                                       
data1 <- read.table("~/scr/EYETRACKING/VSLK/VSLK_LCP/E1_EN_SPR/data/e1_en_spr_data.txt",header=FALSE)
names(data1) <- c("subject", "experiment", "item", "condition", "wordInItem", "word", "SPR_RT")
data1$Response = NA
data1$comma = NA

data2 <- read.table("~/scr/EYETRACKING/VSLK/VSLK_LCP/E5_EN_SPR/data/rawdata.txt",header=FALSE)
names(data2) <- c("subject", "experiment", "item", "condition", "wordInItem", "word", "Response", "SPR_RT")
data2 = data2 %>% mutate(subject = subject+1000)
data2$comma = NA

# Cannot take these, as they have a different tokenization
#data3 <- read.table("~/scr/EYETRACKING/VSLK/VSLK_LCP/E6a6b_EN_SPR/data/e6a_all.txt",header=FALSE)
#names(data3) <- c("comma", "subject", "experiment", "item", "condition", "wordInItem", "word", "Response", "SPR_RT")
#data3 = data3 %>% mutate(subject = subject+2000)
#
#data4 <- read.table("~/scr/EYETRACKING/VSLK/VSLK_LCP/E6a6b_EN_SPR/data/e6b_all.txt",header=FALSE)
#names(data4) <- c("comma", "subject", "experiment", "item", "condition", "wordInItem", "word", "Response", "SPR_RT")
#data4 = data4 %>% mutate(subject = subject+3000)

data = rbind(data1, data2)

print("Number of subjects")
print(length(unique(data$subject)))


data$item_ = ifelse(data$item < 10 & data$experiment == "E1", paste("0", data$item, sep=""), as.character(data$item))

data$VSLK_ID = paste(data$experiment, data$item_, data$condition, sep="_")

data = merge(data, fillers_names, by=c("VSLK_ID"), all.x=TRUE)
print(unique((data[is.na(data$Sentence),])$VSLK_ID))
#[1] "filler_24_3.4.4" "filler_24_4.4.4" "gug_24_a"        "gug_24_b"        "gug_24_c"        "gug_24_d"       


data = data[!is.na(data$Sentence),]


# Match with the Maze data
maze = read.csv("../../../../experiments/maze/experiment1/Submiterator-master/trials_byWord.tsv", sep="\t", quote="@") %>% filter(condition == "filler")


maze_et = merge(maze %>% mutate(sentence=as.character(sentence)) %>% group_by(item, sentence, word, wordInItem) %>% summarise(rt = mean(rt)), data %>% rename(VSLK_item = item) %>% mutate(Sentence=as.character(Sentence)) %>% rename(sentence=Sentence), by=c("sentence", "wordInItem"), all=TRUE)

print("# Only the practice items should be missing")
print(unique((maze_et[is.na(maze_et$condition),])$sentence))
maze_et = maze_et %>% filter(!is.na(condition))
maze_et = maze_et[!is.na(maze_et$word.x),]
#print("# These are the other conditions not included in the Maze study")
#print(unique((maze_et[is.na(maze_et$rt),])$sentence))

print("This should be all TRUE")
print(summary(as.character(maze_et$word.x) == as.character(maze_et$word.y)))
print(head(maze_et[!is.na(maze_et$word.x) && as.character(maze_et$word.x) != as.character(maze_et$word.y),]))
#crash()

maze_et = maze_et %>% filter(!is.na(maze_et$rt))

model$itemID = paste(model$item, model$wordInItem, sep="_")
#sink("analyzeFillers_freq_BNC.R.tsv")
#sink()

#alreadyDone = read.csv("analyzeFillers_freq_EYE_BNC.R.tsv", sep="\t", header=F)$V3

#crash()

# TODO look at n't --> breaks tokenization
sink("output/analyzeFillers_freq_BNC_SPR_Spillover_Averaged_New_R.tsv")
cat("predictability_weight", "deletion_rate", "NData", "AIC", "Coefficient", "Correlation", "\n", sep="\t")
sink()

configs = unique(model %>% select(deletion_rate, predictability_weight))

overall_data = data.frame()


data = maze_et %>% filter(wordInItem != "?")
#data$item = paste("Filler", data$item, sep="_")

human = data %>% mutate(word=word.x) %>% select(sentence, word, wordInItem, item, subject, SPR_RT) %>% filter(!grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word
human = human %>% mutate(item = str_replace(item, "232_", ""))

humanRaw = human
#crash()

for(i in (1:nrow(configs))) {
   if(TRUE) { #!(ID_  %in% alreadyDone)) {
      delta = configs$deletion_rate[[i]]
      lambda = configs$predictability_weight[[i]]
      data = model %>% filter(deletion_rate==delta, predictability_weight==lambda) %>% group_by(wordInItem, item, itemID) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted), WordLength=mean(WordLength), LogWordFreq_COCA.R=mean(LogWordFreq_COCA.R), LogWordFreq=mean(LogWordFreq))
      data = merge(data, maze_et, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq), !is.na(LogWordFreq_COCA.R))
      if(nrow(data) > 0) {
         data$SurprisalReweighted = resid(lm(SurprisalReweighted ~ LogWordFreq + LogWordFreq_COCA.R, data=data))
         data$LogSPRRT = log(data$SPR_RT)
         dataPrevious = data %>% mutate(wordInItem=wordInItem+1)
         data = merge(data, dataPrevious, by=c("wordInItem", "sentence", "subject"))
         lmermodel = lmer(LogSPRRT.x ~ SurprisalReweighted.x + wordInItem + LogWordFreq.x + LogWordFreq_COCA.R.x + WordLength.x + SurprisalReweighted.y + wordInItem + LogWordFreq.y + LogWordFreq_COCA.R.y + WordLength.y + LogSPRRT.y + (1|itemID.x) + (1|subject), data=data, REML=F)
         cat(lambda, delta, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
         # Print result to file
         sink("output/analyzeFillers_freq_BNC_SPR_Spillover_Averaged_New_R.tsv", append=TRUE)
         cat(lambda, delta, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
         sink()
      }
   }
}


