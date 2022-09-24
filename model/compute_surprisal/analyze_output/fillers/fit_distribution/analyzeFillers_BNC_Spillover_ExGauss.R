library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lme4)
library(dplyr)
model = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3_Fillers.py.tsv", sep="\t") %>% mutate(wordInItem = as.numeric(as.character(Region))+1) %>% mutate(item = paste("Filler", Sentence, sep="_")) %>% group_by(item, wordInItem, Region, Word, Script, ID, predictability_weight, deletion_rate) %>% summarise(SurprisalReweighted=mean(as.numeric(as.character(SurprisalReweighted)), na.rm=TRUE)) %>% filter(grepl("_TPLf", Script), predictability_weight==1)

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


      data = model %>% filter(deletion_rate==0.5, predictability_weight==1) %>% group_by(wordInItem, item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE))
      data = merge(data, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq))

library(lme4)

data = data %>% mutate(itemID = paste(item, wordInItem, sep="_"))
data$LogRT = log(data$rt)

library(moments)
exgauss_fit = data %>% group_by(wordInItem, item) %>% mutate(m = mean(rt), gamma1=skewness(rt), s = sd(rt), SurprisalReweighted=mean(SurprisalReweighted))
exgauss_fit = exgauss_fit %>% mutate(mu = m-s*(gamma1/2)^(1/3), sigma2 = s^2 * (1-(gamma1/2)^(2/3)), tau = s*(gamma1/2)^(1/3))

library(ggplot2)
plot = ggplot(exgauss_fit, aes(x=SurprisalReweighted, y=mu)) + geom_smooth() + geom_point() 
ggsave(plot, file="figures/surprisal-rts-exgaussian-plot1.pdf", height=8, width=8)

plot = ggplot(exgauss_fit, aes(x=SurprisalReweighted, y=sqrt(sigma2))) + geom_smooth() + geom_point() 
ggsave(plot, file="figures/surprisal-rts-exgaussian-plot2.pdf", height=8, width=8)

plot = ggplot(exgauss_fit, aes(x=SurprisalReweighted, y=tau)) + geom_smooth() + geom_point() 
ggsave(plot, file="figures/surprisal-rts-exgaussian-plot3.pdf", height=8, width=8)


