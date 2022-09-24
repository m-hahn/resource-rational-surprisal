library(tidyr)
library(dplyr)
data_E0 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E0.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2)
data_E1 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E1.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2)  %>% filter(!(Item %in% c("Mixed_0", "Mixed_10", "Mixed_20", "ItemMixed_22", "ItemMixed_27", "ItemMixed_28", "ItemMixed_30", "ItemMixed_32")))
data_E2 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2) %>% filter(!grepl("245_", Item)) #%>% filter(grepl("_L", Script) | grepl("_TPL", Script))

data = rbind(data_E0, data_E1, data_E2)

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
#plot = ggplot(data %>% filter(Region=="V1_0", Experiment2) %>% group_by(Noun, Condition, deletion_rate, predictability_weight, EmbBias.C) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=EmbBias.C, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(deletion_rate~predictability_weight)
#ggsave(plot, file="analyze_R_Experiment2.pdf", height=5, width=5)

data$compatible = (data$compatible.C>0)
#plot = ggplot(data %>% filter(Region=="V1_0") %>% group_by(Noun, Condition, compatible, SC.C, RC.C, deletion_rate, predictability_weight, EmbBias.C) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=EmbBias.C, y=SurprisalReweighted, group=Condition, color=paste(SC.C,RC.C), linetype=compatible)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate)
#ggsave(plot, file="analyze_R.pdf", height=10, width=10)


library(brms)

for(delta in c(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 )) {
 for(lambda in c(0.0, 0.25, 0.5,  0.75, 1)) {
  outFileName = paste("slopes/", "FIXED_analyze_inter_Fives_Simple_ThatFraction.R_VN5_", delta, "_", lambda, ".tsv", sep="")
#  if(!file.exists(outFileName)) {
   cat(delta, " ", lambda, "\n")
   data2 = data %>% filter(deletion_rate==delta, predictability_weight==lambda) %>% group_by(Noun, compatible.C, EmbBias.C, Item, SC.C, RC.C) %>% summarise(ThatFractionReweighted=median(ThatFractionReweighted))
   if(nrow(data2) > 0) {
#     contrasts(data2$Item) = contr.sum(length(levels(data2$Item)))
     model = lm(ThatFractionReweighted ~ RC.C * compatible.C * Item + EmbBias.C*RC.C * Item + EmbBias.C*compatible.C*Item, data=data2)
     write.table(round(summary(model)$coef,3), file=outFileName, sep="\t")          
  }
}
}
#}

