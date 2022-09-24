library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lme4)
library(dplyr)
model = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3_Fillers.py.tsv", sep="\t") %>% mutate(wordInItem = as.numeric(as.character(Region))+1) %>% mutate(item = paste("Filler", Sentence, sep="_")) %>% group_by(item, wordInItem, Region, Word, Script, ID, predictability_weight, deletion_rate) %>% summarise(SurprisalReweighted=mean(as.numeric(as.character(SurprisalReweighted)), na.rm=TRUE)) %>% filter(grepl("_TP", Script), predictability_weight==1)

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

data1 = read.csv("../../../../experiments/maze/experiment1/Submiterator-master/trials_byWord.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid)
data2 = read.csv("../../../../experiments/maze/experiment2/Submiterator-master/trials-experiment2.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid+1000)
rts = rbind(data1, data2)

rts = rts[rts$rt < quantile(rts$rt, 0.99),]
rts = rts[rts$rt > quantile(rts$rt, 0.01),]
rts = rts %>% mutate(item = str_replace(item, "232_", ""))
rts = rts %>% filter(!grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word
rts = rts %>% select(wordInItem, item, workerid, rt, word) %>% filter(!grepl("Mixed", item), !grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word
rts = rts %>% mutate(item = str_replace(item, "232_", ""))



model$itemID = paste(model$item, model$wordInItem, sep="_")
#sink("analyzeFillers_freq_BNC.R.tsv")
#sink()

#alreadyDone = read.csv("analyzeFillers_freq_BNC.R.tsv", sep="\t", header=F)$V3

##################################
# With 0.05
data = model %>% filter(deletion_rate==0.05, predictability_weight==1) %>% group_by(wordInItem, item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE))
data = merge(data, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq))


model_critical = read.csv("../../../../experiments/maze/meta/output/analyze_Model_PlotForExpt2_Joint.R_Expt1.tsv", sep="\t") %>% filter(deletion_rate==0.05, predictability_weight==1) %>% select(rt, SurprisalReweighted, HasRC, HasSC, compatible)

data_fillers = data %>% group_by(wordInItem, item) %>% summarise(rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)) %>% group_by() %>% select(rt, SurprisalReweighted) %>% mutate(HasRC=NA, HasSC=NA, compatible=NA)

data_merged = rbind(model_critical, data_fillers)

library(ggplot2)
plot = ggplot(data_fillers, aes(x=SurprisalReweighted, y=rt)) + geom_smooth() + geom_point(alpha=0.5) + geom_point(data=model_critical, aes(color=paste(HasSC, HasRC), type=compatible))
ggsave(plot, file="figures/surprisal-rts-plot1-model005-expt1.pdf", height=10, width=10)




##################################
# With 0.5
data = model %>% filter(deletion_rate==0.5, predictability_weight==1) %>% group_by(wordInItem, item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE))
data = merge(data, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq))


model_critical = read.csv("../../../../experiments/maze/meta/output/analyze_Model_PlotForExpt2_Joint.R_Expt1.tsv", sep="\t") %>% filter(deletion_rate==0.5, predictability_weight==1) %>% select(rt, SurprisalReweighted, HasRC, HasSC, compatible)

data_fillers = data %>% group_by(wordInItem, item) %>% summarise(rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)) %>% group_by() %>% select(rt, SurprisalReweighted) %>% mutate(HasRC=NA, HasSC=NA, compatible=NA)

data_merged = rbind(model_critical, data_fillers)

library(ggplot2)
plot = ggplot(data_fillers, aes(x=SurprisalReweighted, y=rt)) + geom_smooth() + geom_point(alpha=0.5) + geom_point(data=model_critical, aes(color=paste(HasSC, HasRC), type=compatible))
ggsave(plot, file="figures/surprisal-rts-plot1-model05-expt1.pdf", height=10, width=10)



##################################
# With 0.05
data = model %>% filter(deletion_rate==0.05, predictability_weight==1) %>% group_by(wordInItem, item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE))
data = merge(data, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq))


model_critical = read.csv("../../../../experiments/maze/meta/output/analyze_Model_PlotForExpt2_Joint.R_Expt2.tsv", sep="\t") %>% filter(deletion_rate==0.05, predictability_weight==1) %>% select(rt, SurprisalReweighted, HasRC, HasSC, compatible)

data_fillers = data %>% group_by(wordInItem, item) %>% summarise(rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)) %>% group_by() %>% select(rt, SurprisalReweighted) %>% mutate(HasRC=NA, HasSC=NA, compatible=NA)

data_merged = rbind(model_critical, data_fillers)

library(ggplot2)
plot = ggplot(data_fillers, aes(x=SurprisalReweighted, y=rt)) + geom_smooth() + geom_point(alpha=0.5) + geom_point(data=model_critical, aes(color=paste(HasSC, HasRC), type=compatible))
ggsave(plot, file="figures/surprisal-rts-plot1-model005-expt2.pdf", height=10, width=10)




##################################
# With 0.5
data = model %>% filter(deletion_rate==0.5, predictability_weight==1) %>% group_by(wordInItem, item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE))
data = merge(data, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq))


model_critical = read.csv("../../../../experiments/maze/meta/output/analyze_Model_PlotForExpt2_Joint.R_Expt2.tsv", sep="\t") %>% filter(deletion_rate==0.5, predictability_weight==1) %>% select(rt, SurprisalReweighted, HasRC, HasSC, compatible)

data_fillers = data %>% group_by(wordInItem, item) %>% summarise(rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)) %>% group_by() %>% select(rt, SurprisalReweighted) %>% mutate(HasRC=NA, HasSC=NA, compatible=NA)

data_merged = rbind(model_critical, data_fillers)

library(ggplot2)
plot = ggplot(data_fillers, aes(x=SurprisalReweighted, y=rt)) + geom_smooth() + geom_point(alpha=0.5) + geom_point(data=model_critical, aes(color=paste(HasSC, HasRC), type=compatible))
ggsave(plot, file="figures/surprisal-rts-plot1-model05-expt2.pdf", height=10, width=10)



