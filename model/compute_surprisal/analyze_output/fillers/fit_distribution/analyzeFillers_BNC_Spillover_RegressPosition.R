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
lmer_model = (lmer(LogRT ~ SurprisalReweighted * wordInItem + (1|workerid) + (1|itemID), data=data))

summary(lmer_model)
#                                 Estimate Std. Error t value
#(Intercept)                     6.6144506  0.0200182 330.422
#SurprisalReweighted             0.0280486  0.0020562  13.641
#wordInItem                     -0.0046842  0.0014979  -3.127
#SurprisalReweighted:wordInItem  0.0015747  0.0002528   6.229


#lmer_model = (lmer(LogRT ~ LogWordFreq + SurprisalReweighted * wordInItem + (1|workerid) + (1|itemID), data=data))

#                                Estimate Std. Error t value
#(Intercept)                     6.758546   0.048787 138.530
#LogWordFreq                    -0.015718   0.004857  -3.236
#SurprisalReweighted             0.023661   0.002452   9.650
#WordInItem                     -0.005153   0.001495  -3.446
#SurprisalReweighted:wordInItem  0.001716   0.000255   6.732

library(ggplot2)
plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise(rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=SurprisalReweighted, y=rt)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~wordInItem)
ggsave(plot, file="figures/surprisal-rts-plot1-byPosition.pdf", height=15, width=15)


