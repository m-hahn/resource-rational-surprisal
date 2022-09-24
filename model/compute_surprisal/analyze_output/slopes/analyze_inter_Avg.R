library(tidyr)
library(dplyr)
data_E1 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E1.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2)
data_E2 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2) %>% filter(!grepl("245_", Item))

data = rbind(data_E1, data_E2)

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
plot = ggplot(data %>% filter(Region=="V1_0") %>% group_by(Noun, Condition, compatible, SC.C, RC.C, deletion_rate, predictability_weight, EmbBias.C) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=EmbBias.C, y=SurprisalReweighted, group=Condition, color=paste(SC.C,RC.C), linetype=compatible)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate)
ggsave(plot, file="analyze_R.pdf", height=10, width=10)


library(brms)
if(TRUE){
configurations = unique(data %>% select(deletion_rate, predictability_weight))
configurations = configurations[sample(1:nrow(configurations)),]
   data2 = data %>% filter(deletion_rate>0.33, deletion_rate<0.66, predictability_weight>=0.5) %>% group_by(Noun, compatible.C, EmbBias.C, Item, RC.C) %>% filter(SC.C>0) %>% summarise(SurprisalReweighted=median(SurprisalReweighted))
   model = brm(SurprisalReweighted ~ RC.C * compatible.C + EmbBias.C*RC.C + EmbBias.C*compatible.C + (1+compatible.C+EmbBias.C+RC.C * compatible.C + EmbBias.C*RC.C + EmbBias.C*compatible.C|Item) + (1|Noun), data=data2, cores=4, iter=5000)
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
                                                                                                                                 
   write.table(summary(model)$fixed, file=paste("slopes/", "FIXED_analyze_inter_Avg.R_VN5.tsv", sep=""), sep="\t")          
   write.csv(slopes,file=paste("slopes/", "analyze_inter_Avg.R_VN5.tsv", sep=""))
   model=NULL
   samples=NULL
   cleanMem <- function(n=10) { for (i in 1:n) gc() }
}


