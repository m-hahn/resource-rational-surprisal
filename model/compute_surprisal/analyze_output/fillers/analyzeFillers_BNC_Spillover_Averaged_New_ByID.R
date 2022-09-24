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
data1 = read.csv("../../../../experiments/maze/experiment1/Submiterator-master/trials_byWord.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid, distractor=NA, group=NA)
data2 = read.csv("../../../../experiments/maze/experiment2/Submiterator-master/trials-experiment2.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid+1000, distractor=NA, group=NA)
data3 = read.csv("../../../../experiments/maze/previous/study5_replication/Submiterator-master/all_trials.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid+2000, distractor=NA, group=NA)
rts = rbind(data1, data2, data3)
#crash()

rts = rts[rts$rt < quantile(rts$rt, 0.99),]
rts = rts[rts$rt > quantile(rts$rt, 0.01),]
rts = rts %>% mutate(item = str_replace(item, "232_", ""))
rts = rts %>% filter(!grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word
rts = rts %>% select(wordInItem, item, workerid, rt) %>% filter(!grepl("Mixed", item), !grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word
rts = rts %>% mutate(item = str_replace(item, "232_", ""))



model$itemID = paste(model$item, model$wordInItem, sep="_")
#sink("analyzeFillers_freq_BNC.R.tsv")
#sink()

#alreadyDone = read.csv("analyzeFillers_freq_BNC.R.tsv", sep="\t", header=F)$V3


# TODO look at n't --> breaks tokenization
sink("analyzeFillers_freq_BNC_Spillover_Averaged_New_ByID_R.tsv")
cat("predictability_weight", "deletion_rate", "ID", "NData", "AIC", "Coefficient", "Correlation", "\n", sep="\t")
sink()

configs = unique(model %>% select(deletion_rate, predictability_weight, ID))
for(i in (1:nrow(configs))) {
   if(TRUE) { #!(ID_  %in% alreadyDone)) {
      delta = configs$deletion_rate[[i]]
      lambda = configs$predictability_weight[[i]]
      id_ = configs$ID[[i]]
      data = model %>% filter(ID==id_, deletion_rate==delta, predictability_weight==lambda) %>% group_by(wordInItem, item, itemID) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted), WordLength=mean(WordLength), LogWordFreq_COCA.R=mean(LogWordFreq_COCA.R), LogWordFreq=mean(LogWordFreq))
      data = merge(data, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq), !is.na(LogWordFreq_COCA.R), !is.na(SurprisalReweighted))
      if(nrow(data) > 100) {
         data$SurprisalReweighted = resid(lm(SurprisalReweighted ~ LogWordFreq + LogWordFreq_COCA.R, data=data))
         data$LogRT = log(data$rt)
         dataPrevious = data %>% mutate(wordInItem=wordInItem+1)
         data = merge(data, dataPrevious, by=c("wordInItem", "item", "workerid")) # "sentence",
#         lmermodel = lmer(LogRT.x ~ (1|itemID.x), data=data)
         if(nrow(data) > 100) {
           lmermodel = lmer(LogRT.x ~ SurprisalReweighted.x + wordInItem + LogWordFreq.x + LogWordFreq_COCA.R.x + WordLength.x + SurprisalReweighted.y + LogWordFreq.y + LogWordFreq_COCA.R.y + WordLength.y + LogRT.y + (1|itemID.x) + (1|workerid), data=data, REML=F)
           cat(lambda, delta, id_, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
           sink("analyzeFillers_freq_BNC_Spillover_Averaged_New_ByID_R.tsv", append=TRUE)
           cat(lambda, delta, id_, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
           sink()
         }
      }
   }
}

