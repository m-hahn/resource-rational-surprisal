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
                                                                                                                       
data <- read.table("~/scr/EYETRACKING/VSLK/VSLK_LCP/E2_EN_ET/data/e2_en_et_ncdata.txt",header=TRUE)
#data <- read.table("~/scr/EYETRACKING/VSLK/VSLK_LCP/E2_EN_ET/data/e2_en_et_data.txt",header=TRUE)   

data$item_ = ifelse(data$item < 10 & data$experiment == "E1", paste("0", data$item, sep=""), as.character(data$item))

data$VSLK_ID = paste(data$experiment, data$item_, data$condition, sep="_")

data = merge(data, fillers_names, by=c("VSLK_ID"), all.x=TRUE)
#> unique((data[is.na(data$Sentence),])$VSLK_ID)
#[1] "filler_24_3.4.4" "filler_24_4.4.4" "gug_24_a"        "gug_24_b"        "gug_24_c"        "gug_24_d"       

data = data[!is.na(data$Sentence),]


# Match with the Maze data
maze = read.csv("../../../../experiments/maze/experiment1/Submiterator-master/trials_byWord.tsv", sep="\t", quote="@") %>% filter(condition == "filler")


maze_et = merge(maze %>% group_by(item, sentence, word, wordInItem) %>% summarise(rt = mean(rt)), data %>% mutate(wordInItem = roi-1) %>% rename(VSLK_item = item) %>% rename(sentence=Sentence), by=c("sentence", "wordInItem"), all=TRUE)

print("# Only the practice items should be missing")
print(unique((maze_et[is.na(maze_et$condition),])$sentence))
maze_et = maze_et %>% filter(!is.na(condition))

print("# These are the other conditions not included in the Maze study")
print(unique((maze_et[is.na(maze_et$rt) & !is.na(maze_et$RBRT),])$sentence))

maze_et = maze_et %>% filter(!is.na(maze_et$rt), !is.na(maze_et$RBRT))

model$itemID = paste(model$item, model$wordInItem, sep="_")
#sink("analyzeFillers_freq_BNC.R.tsv")
#sink()

#alreadyDone = read.csv("analyzeFillers_freq_EYE_BNC.R.tsv", sep="\t", header=F)$V3

#crash()

# TODO look at n't --> breaks tokenization
sink("output/analyzeFillers_freq_BNC_TFT_Spillover_Averaged_New_R.tsv")
cat("predictability_weight", "deletion_rate", "NData", "AIC", "Coefficient", "Correlation", "\n", sep="\t")
sink()

configs = unique(model %>% select(deletion_rate, predictability_weight))

overall_data = data.frame()


data = maze_et %>% filter(wordInItem != "?")
#data$item = paste("Filler", data$item, sep="_")

human = data %>% select(sentence, word, wordInItem, item, subject, TFT, TFT, RRT) %>% filter(!grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word
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
         data$ReadingTime = data$TFT
         dataPrevious = data %>% mutate(wordInItem=wordInItem+1)
         data = merge(data, dataPrevious, by=c("wordInItem", "sentence", "subject"))
         lmermodel = lmer(ReadingTime.x ~ SurprisalReweighted.x + wordInItem + LogWordFreq.x + LogWordFreq_COCA.R.x + WordLength.x + SurprisalReweighted.y + LogWordFreq.y + LogWordFreq_COCA.R.y + WordLength.y + ReadingTime.y + (1|itemID.x) + (1|subject), data=data %>% filter(TFT.x > 0))
         cat(lambda, delta, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
         # Print result to file
         sink("output/analyzeFillers_freq_BNC_TFT_Spillover_Averaged_New_R.tsv", append=TRUE)
         cat(lambda, delta, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
         sink()
      }
   }
}
#
#human$itemID =as.numeric(as.factor( paste(human$item, human$wordInItem, sep="_")))
#itemID = human$itemID
#ReadingTime_TFT = log(human$TFT)
#ReadingTime_RRT = log(human$RRT)
#ReadingTime_TFT = log(human$TFT)
#
#subject = as.numeric(as.factor(human$subject))
##crash()
#
#human$itemID = NULL
#human$rt = NULL
#human$subject = NULL
#human$wordInItem = NULL
#human$item = NULL
#human$sentence=NULL
#human$word=NULL
#human$TFT=NULL
#human$RRT=NULL
#human$TFT=NULL
#write.table(ReadingTime_TFT, file="forStan_Fillers_EYE/ReadingTime_TFT.tsv", quote=F, sep="\t")
#write.table(ReadingTime_RRT, file="forStan_Fillers_EYE/ReadingTime_RRT.tsv", quote=F, sep="\t")
#write.table(ReadingTime_TFT, file="forStan_Fillers_EYE/ReadingTime_TFT.tsv", quote=F, sep="\t")
#write.table(human, file="forStan_Fillers_EYE/predictions.tsv", quote=F, sep="\t")
#write.table(subject, file="forStan_Fillers_EYE/subjects.tsv", quote=F, sep="\t")
#write.table(itemID, file="forStan_Fillers_EYE/items.tsv", quote=F, sep="\t")
#
#
#
#
#
