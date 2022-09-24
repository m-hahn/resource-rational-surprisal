library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lme4)
library(dplyr)
model = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3_Fillers.py.tsv", sep="\t") %>% mutate(predictability_weight = as.numeric(as.character(predictability_weight)), SurprisalReweighted=as.numeric(as.character(SurprisalReweighted)), deletion_rate=as.numeric(as.character(deletion_rate)), Region=as.numeric(as.character(Region)), Sentence=as.numeric(as.character(Sentence)), wordInItem = as.numeric(as.character(Region))+1) %>% mutate(item = paste("Filler", Sentence, sep="_")) %>% group_by(item, wordInItem, Region, Word, Script, ID, predictability_weight, deletion_rate) %>% summarise(SurprisalReweighted=mean(as.numeric(as.character(SurprisalReweighted)), na.rm=TRUE)) %>% filter(grepl("_TPLf", Script), predictability_weight==1)

model$LowerCaseToken = tolower(model$Word)
model$WordLength = nchar(as.character(model$Word))


data1 = read.csv("../../../../experiments/maze/experiment1/Submiterator-master/trials_byWord.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid)
data2 = read.csv("../../../../experiments/maze/experiment2/Submiterator-master/trials-experiment2.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid+1000)
rts = rbind(data1, data2)

rts = rts[rts$rt < quantile(rts$rt, 0.99),]
rts = rts[rts$rt > quantile(rts$rt, 0.01),]


model$itemID = paste(model$item, model$wordInItem, sep="_")
#sink("analyzeFillers_freq_BNC.R.tsv")
#sink()

#alreadyDone = read.csv("analyzeFillers_freq_BNC.R.tsv", sep="\t", header=F)$V3


# TODO look at n't --> breaks tokenization
configs = unique(model %>% group_by() %>% select(deletion_rate, predictability_weight))

overall_data = data.frame()


human = rts %>% select(wordInItem, item, workerid, rt) %>% filter(!grepl("Mixed", item), !grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word
human = human %>% mutate(item = str_replace(item, "232_", ""))

humanRaw = human
#crash()
# Go through all cells
for(i in (1:nrow(configs))) {
   delta = configs$deletion_rate[[i]];    lambda = configs$predictability_weight[[i]]
   cat(i, nrow(configs), delta, lambda, "\n")
   relevant = model %>% filter(abs(deletion_rate - delta) <= 0.0, abs(predictability_weight - lambda) <= 0.0) %>% group_by(wordInItem, item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE))
   relevant[[paste("Surprisal_", delta, "_", lambda, sep="")]] = relevant$SurprisalReweighted
   relevant$SurprisalReweighted = NULL
   human = merge(relevant, human, by=c("wordInItem", "item"), all.y=TRUE)
   if(any(is.na(human[[paste("Surprisal_", delta, "_", lambda, sep="")]]))) {
      cat("Warning NAs!\n")
      human[[paste("Surprisal_", delta, "_", lambda, sep="")]] = NULL
   }
}

human$itemID =as.numeric(as.factor( paste(human$item, human$wordInItem, sep="_")))
itemID = human$itemID
rt = human$rt
LogRT = log(human$rt)

workerid = as.numeric(as.factor(human$workerid))
#crash()

human$itemID = NULL
human$rt = NULL
human$LogRT = NULL
human$workerid = NULL
human$wordInItem = NULL
human$item = NULL
write.table(LogRT, file="forStan_Fillers/LogRT.tsv", quote=F, sep="\t")
write.table(human, file="forStan_Fillers/predictions.tsv", quote=F, sep="\t")
write.table(workerid, file="forStan_Fillers/subjects.tsv", quote=F, sep="\t")
write.table(itemID, file="forStan_Fillers/items.tsv", quote=F, sep="\t")





