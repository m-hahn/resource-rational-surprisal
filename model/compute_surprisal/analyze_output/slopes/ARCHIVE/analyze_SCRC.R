library(tidyr)
library(dplyr)
data = read.csv("/juice/scr/mhahn/TMP/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN4.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2)
data0 = read.csv("/juice/scr/mhahn/TMP/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN4Stims_3_W_GPT2M_ZERO.py_234877849_ZeroLoss", sep="\t") %>% mutate(Word = NA, Script='ZERO', ID='ZERO', predictability_weight=0.5, deletion_rate=0, autoencoder="NONE", lm="NONE")

data=rbind(data, data0) %>% filter(!grepl("participle", Item))

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


library(ggplot2)
plot = ggplot(data %>% filter(Region=="V1_0", Experiment2) %>% group_by(Noun, Condition, deletion_rate, predictability_weight, EmbBias.C) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=EmbBias.C, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(deletion_rate~predictability_weight)
ggsave(plot, file="analyze_R_Experiment2.pdf", height=5, width=5)

plot = ggplot(data %>% filter(Region=="V1_0", !Experiment2) %>% group_by(Noun, Condition, deletion_rate, predictability_weight, EmbBias.C) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=EmbBias.C, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(deletion_rate~predictability_weight)
ggsave(plot, file="analyze_R_Experiment1.pdf", height=5, width=5)


library(brms)
if(TRUE){
configurations = unique(data %>% select(deletion_rate, predictability_weight))
for(i in (1:nrow(configurations))) {
   delta = configurations$deletion_rate[[i]]
   lambda = configurations$predictability_weight[[i]]
   data2 = data %>% filter(deletion_rate==delta, predictability_weight==lambda) %>% group_by(Noun, compatible.C, EmbBias.C, Item, RC.C) %>% filter(SC.C>0) %>% summarise(SurprisalReweighted=median(SurprisalReweighted))
   model = brm(SurprisalReweighted ~ compatible.C + EmbBias.C + (1+compatible.C+EmbBias.C|Item) + (1+compatible.C|Noun), data=data2 %>% filter(RC.C>0), cores=4, iter=10000)
   # now get and output the per-item slopes
   samples = posterior_samples(model)
   slopes = data.frame(Item=c(), EmbBias=c(), Compatible=c())
   library(stringr)
   for(item_ in unique(data2$Item)) {
      item__ = str_replace_all(item_, " ", ".")
      slope_embBias = mean(samples$b_EmbBias.C + samples[[paste("r_Item[", item__, ",EmbBias.C]", sep="")]])
      slope_compat = mean(samples$b_compatible.C + samples[[paste("r_Item[", item__, ",compatible.C]", sep="")]])
      slopes = rbind(slopes, data.frame(Item=c(item_), EmbBias=c(slope_embBias), Compatible=c(slope_compat)))

   }
                                                                                                                                 
   write.table(summary(model)$fixed, file=paste("slopes/", "FIXED_analyze_SCRC.R", delta, "_", lambda, ".tsv", sep=""), sep="\t")          
   write.csv(slopes,file=paste("slopes/", "analyze_SCRC.R", delta, "_", lambda, ".tsv", sep=""))
   model=NULL
   samples=NULL
   cleanMem <- function(n=10) { for (i in 1:n) gc() }
}
}


