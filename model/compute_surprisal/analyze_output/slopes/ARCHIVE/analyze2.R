library(tidyr)
library(dplyr)
data = read.csv("/juice/scr/mhahn/TMP/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN5.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2)
data0 = read.csv("/juice/scr/mhahn/TMP/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN5Stims_3_W_GPT2M_ZERO.py_234877849_ZeroLoss", sep="\t") %>% mutate(Word = NA, Script='ZERO', ID='ZERO', predictability_weight=0.5, deletion_rate=0, autoencoder="NONE", lm="NONE")

data=rbind(data, data0)

nounFreqs = read.csv("/u/scr/mhahn/CODE/forgetting/corpus_counts/wikipedia/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)


nounFreqs2 = read.csv("/u/scr/mhahn/CODE/forgetting/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs %>% rename(Noun = noun), by=c("Noun"), all.x=TRUE)

data = data %>% mutate(EmbBias.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))

print(unique((data %>% filter(is.na(EmbBias.C)))$Noun))
# [1] conjecture  guess       insinuation intuition   observation

data$compatible.C = (grepl("_co", data$Condition)-0.5)
data$RC.C = (grepl("SCRC", data$Condition)-0.5)
data$SC.C = (0.5-grepl("NoSC", data$Condition))

data[data$SC.C < 0,]$compatible.C = 0
data[data$SC.C < 0,]$RC.C = 0


data$Item245 = grepl("245_", data$Item)

data$Adv = grepl("adv", data$Item)
data = data %>% filter(!Adv)

data$Experiment2 = grepl("_Critical_", data$Item)
data$Experiment2.C = data$Experiment2 - mean(data$Experiment2)

library(lme4)
library(brms)
#model = lmer(SurprisalReweighted ~ RC.C + Experiment2.C*compatible.C + Experiment2.C*EmbBias.C + (1|Item) + (1|Noun), data=data %>% filter(deletion_rate==0, Region == "V1_0", SC.C>0))
#model = lmer(SurprisalReweighted ~ RC.C + Experiment2.C*compatible.C + Experiment2.C*EmbBias.C + (1|ID) + (1|Item) + (1|Noun), data=data %>% filter(deletion_rate>0, Region == "V1_0", SC.C>0))
#model = lmer(SurprisalReweighted ~ RC.C + Experiment2.C*compatible.C + Experiment2.C*EmbBias.C + (1|Item) + (1|Noun), data=data %>% filter(deletion_rate>0, Region == "V1_0", SC.C>0) %>% group_by(RC.C, Experiment2.C, compatible.C, EmbBias.C, Noun, Item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE)))
#model = brm(SurprisalReweighted ~ RC.C + Experiment2.C*compatible.C + Experiment2.C*EmbBias.C + (1+RC.C+compatible.C+EmbBias.C|Item) + (1+RC.C+compatible.C|Noun), data=data %>% filter(deletion_rate==0, Region == "V1_0", SC.C>0))
#model = lmer(SurprisalReweighted ~ RC.C + Experiment2.C*compatible.C + Experiment2.C*EmbBias.C + (1+RC.C+compatible.C+EmbBias.C|Item) + (1+RC.C+compatible.C|Noun), data=data %>% filter(deletion_rate==delta, Region == "V1_0", SC.C>0))
#model = lmer(SurprisalReweighted ~ RC.C + Experiment2.C*compatible.C + Experiment2.C*EmbBias.C + (1+RC.C+compatible.C+EmbBias.C|ID) + (1+RC.C+compatible.C+EmbBias.C|Item) + (1+RC.C+compatible.C|Noun), data=data %>% filter(deletion_rate>0, Region == "V1_0", SC.C>0))
#model = brm(SurprisalReweighted ~ RC.C + Experiment2.C*compatible.C + Experiment2.C*EmbBias.C + (1+RC.C+compatible.C+EmbBias.C|ID) + (1+RC.C+compatible.C+EmbBias.C|Item) + (1+RC.C+compatible.C|Noun), data=data %>% filter(deletion_rate>0, Region == "V1_0", SC.C>0), cores=4)

library(brms)

data2 = data %>% filter(deletion_rate>0) %>% group_by(Noun, compatible.C, EmbBias.C, Item, RC.C) %>% filter(SC.C>0) %>% summarise(SurprisalReweighted=median(SurprisalReweighted))


data2_co = data %>% filter(deletion_rate>0, predictability_weight>=0.750, compatible.C>0) %>% group_by(Item) %>% filter(RC.C>0) %>% summarise(SurprisalReweighted_Co=median(SurprisalReweighted))
data2_in = data %>% filter(deletion_rate>0, predictability_weight>=0.750, compatible.C<0) %>% group_by(Item) %>% filter(RC.C>0) %>% summarise(SurprisalReweighted_Inco=median(SurprisalReweighted))

data2 = merge(data2_co, data2_in, by=c("Item")) %>% mutate(CompEffect = SurprisalReweighted_Co-SurprisalReweighted_Inco)

data2 = data %>% filter(deletion_rate>0, predictability_weight==1, SC.C>0) %>% group_by(Item, EmbBias.C) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))


model = lmer(SurprisalReweighted ~ EmbBias.C + (1+EmbBias.C|Item), data=data2)

