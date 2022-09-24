library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data1 = read.csv("../previous/study5_replication/Submiterator-master/all_trials.tsv", sep="\t") %>% mutate(distractor=NA, group=NULL)
data2 = read.csv("../experiment1/Submiterator-master/trials_byWord.tsv", sep="\t") %>% mutate(workerid=workerid+1000)
data3 = read.csv("../experiment2/Submiterator-master/trials-experiment2.tsv", sep="\t") %>% mutate(workerid=workerid+2000)

data = rbind(data1, data2, data3)

# Exclude participants with excessive error rates
participantsByErrorsBySlide = data %>% filter(correct != "none") %>% group_by(workerid) %>% summarise(ErrorsBySlide = mean(correct == "no"))
data = merge(data, participantsByErrorsBySlide, by=c("workerid"))
data = data %>% filter(ErrorsBySlide < 0.2)

# Only consider critical trials
data = data %>% filter(condition != "filler")

# Remove trials with incorrect responses
data = data %>% filter(rt > 0, correct == "yes")

# Remove extremely low or extremely high reading times
data = data %>% filter(rt < quantile(data$rt, 0.99))
data = data %>% filter(rt > quantile(data$rt, 0.01))


# Load corpus counts (Wikipedia)
nounFreqs = read.csv("../../../materials/nouns/corpus_counts/wikipedia/results/results_counts4NEW.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)

nounFreqs2 = read.csv("../../../materials/nouns/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)

# Code Embedding Bias
data = data %>% mutate(EmbeddingBias = True_False_False-False_False_False)
data = data %>% mutate(EmbeddingBias.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))

# Code Conditions
data = data %>% mutate(compatible = ifelse(condition %in% c("critical_SCRC_compatible", "critical_compatible"), TRUE, FALSE))
data = data %>% mutate(HasSC = ifelse(condition == "critical_NoSC", FALSE, TRUE))
data = data %>% mutate(HasRC = ifelse(condition %in% c("critical_SCRC_compatible", "critical_SCRC_incompatible"), TRUE, FALSE))

# Contrast Coding
data$HasRC.C = ifelse(data$HasRC, 0.5, ifelse(data$HasSC, -0.5, 0)) #resid(lm(HasRC ~ HasSC, data=data))
data$HasSC.C = data$HasSC - 0.8
data$compatible.C = ifelse(!data$HasSC, 0, ifelse(data$compatible, 0.5, -0.5))
# Log-Transform Reading Times
data$LogRT = log(data$rt)

# Center trial order
data$trial = data$trial - mean(data$trial, na.rm=TRUE)

            
model_E0 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E0.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2) %>% filter(grepl("_TPL", Script))    
model_E1 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E1.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2) %>% filter(!(Item %in% c("Mixed_0", "Mixed_10", "Mixed_20", "ItemMixed_22", "ItemMixed_27", "ItemMixed_28", "ItemMixed_30", "ItemMixed_32"))) %>% filter(grepl("_TPL", Script))  
model_E2 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2) %>% filter(!grepl("245_", Item)) %>% filter(grepl("_TPL", Script))   
 
model = rbind(model_E0, model_E1, model_E2)   

model$item = model$Item


   
   model_ = model %>% filter(Region == "V1_0", predictability_weight == 1) %>% group_by(predictability_weight, deletion_rate, Noun, item, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE))
   model_$HasSC = !grepl("NoSC", model_$Condition)
   model_$HasRC = grepl("RC", model_$Condition)
   model_$compatible = grepl("_co", model_$Condition)
   


############### For Experiment 2

   data_ = merge(data %>% filter(Region == "REGION_3_0"), model_ %>% rename(noun=Noun), by=c("noun", "item", "HasSC", "HasRC", "compatible"), all.x=TRUE)
  


data_$Experiment = ifelse(data_$workerid<1000, "E0", ifelse(data_$workerid<2000, "E1", "E2"))
 

data_ %>% filter(is.na(data_$Condition), Experiment == "E1")
# this should be empty
data_ %>% filter(is.na(data_$Condition), Experiment == "E2")
# This should only include 238_Critical_VAdv1 trials


data_ = data_ %>% filter(!is.na(data_$Condition))
data_ = data_ %>% filter(Experiment == "E2")

cor.test(data_$LogRT, data_$SurprisalReweighted)


library(ggplot2)
data__ = data_ %>% group_by(deletion_rate, predictability_weight, condition, noun, Condition, HasRC, HasSC, compatible, EmbeddingBias) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogRT=mean(LogRT, na.rm=TRUE), rt=mean(rt, na.rm=TRUE))

plot = ggplot(data__, aes(x=EmbeddingBias, y=SurprisalReweighted, group=Condition, color=paste(HasSC, HasRC))) + geom_point(alpha=0.3) + geom_smooth(method="lm", aes(linetype=compatible))
ggsave(plot, file="figures/analyze_Model_PlotForExpt2_R.pdf", width=4, height=4)

plot = ggplot(data__, aes(x=EmbeddingBias, y=LogRT, group=Condition, color=paste(HasSC, HasRC))) + geom_point(alpha=0.3) + geom_smooth(method="lm", aes(linetype=compatible))
ggsave(plot, file="figures/analyze_Model_PlotForExpt2_Human_R.pdf", width=4, height=4)

plot = ggplot(data__, aes(x=LogRT, y=SurprisalReweighted, group=Condition, color=paste(HasSC, HasRC))) + geom_point(aes(shape=compatible)) + theme_classic() + facet_grid(~deletion_rate) + ylab("Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/analyze_Model_PlotForExpt2_Joint_ModelHuman_R.pdf", width=15, height=1.5)

cor.test(data__$LogRT, data__$SurprisalReweighted)

write.table(data__ %>% filter(deletion_rate %in% c(0.05, 0.5)), file="output/analyze_Model_PlotForExpt2_Joint.R_Expt2.tsv", sep="\t")


plot = ggplot(data__ %>% group_by(deletion_rate) %>% summarise(Correlation = cor(LogRT, SurprisalReweighted)), aes(x=deletion_rate, y=1, fill=Correlation)) + geom_tile()  + theme_classic() + scale_fill_gradient2() + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_PlotForExpt2_Joint_ModelHuman_Corrs_R.pdf", width=15, height=1.5)


############### Same as above, for Experiment 1

   data_ = merge(data %>% filter(Region == "REGION_3_0"), model_ %>% rename(noun=Noun), by=c("noun", "item", "HasSC", "HasRC", "compatible"), all.x=TRUE)
  
data_$Experiment = ifelse(data_$workerid<1000, "E0", ifelse(data_$workerid<2000, "E1", "E2"))
 

data_ %>% filter(is.na(data_$Condition), Experiment == "E1")
# this should be empty
data_ %>% filter(is.na(data_$Condition), Experiment == "E2")
# This should only include 238_Critical_VAdv1 trials


data_ = data_ %>% filter(!is.na(data_$Condition))
data_ = data_ %>% filter(Experiment == "E1")

cor.test(data_$LogRT, data_$SurprisalReweighted)


library(ggplot2)
data__ = data_ %>% group_by(deletion_rate, predictability_weight, condition, noun, Condition, HasRC, HasSC, compatible, EmbeddingBias) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogRT=mean(LogRT, na.rm=TRUE), rt=mean(rt, na.rm=TRUE))

plot = ggplot(data__, aes(x=EmbeddingBias, y=SurprisalReweighted, group=Condition, color=paste(HasSC, HasRC))) + geom_point(alpha=0.3) + geom_smooth(method="lm", aes(linetype=compatible))
ggsave(plot, file="figures/analyze_Model_PlotForExpt1_R.pdf", width=4, height=4)

plot = ggplot(data__, aes(x=EmbeddingBias, y=LogRT, group=Condition, color=paste(HasSC, HasRC))) + geom_point(alpha=0.3) + geom_smooth(method="lm", aes(linetype=compatible))
ggsave(plot, file="figures/analyze_Model_PlotForExpt1_Human_R.pdf", width=4, height=4)

plot = ggplot(data__, aes(x=LogRT, y=SurprisalReweighted, group=Condition, color=paste(HasSC, HasRC))) + geom_point(aes(shape=compatible)) + theme_classic() + facet_grid(~deletion_rate) + ylab("Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/analyze_Model_PlotForExpt1_Joint_ModelHuman_R.pdf", width=15, height=1.5)

cor.test(data__$LogRT, data__$SurprisalReweighted)

write.table(data__ %>% filter(deletion_rate %in% c(0.05, 0.5)), file="output/analyze_Model_PlotForExpt2_Joint.R_Expt1.tsv", sep="\t")


plot = ggplot(data__ %>% group_by(deletion_rate) %>% summarise(Correlation = cor(LogRT, SurprisalReweighted)), aes(x=deletion_rate, y=1, fill=Correlation)) + geom_tile()  + theme_classic() + scale_fill_gradient2() + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_PlotForExpt1_Joint_ModelHuman_Corrs_R.pdf", width=15, height=1.5)


############### Both Experiments

   data_ = merge(data %>% filter(Region == "REGION_3_0"), model_ %>% rename(noun=Noun), by=c("noun", "item", "HasSC", "HasRC", "compatible"), all.x=TRUE)
  


data_$Experiment = ifelse(data_$workerid<1000, "E0", ifelse(data_$workerid<2000, "E1", "E2"))
 

data_ %>% filter(is.na(data_$Condition), Experiment == "E1")
# this should be empty
data_ %>% filter(is.na(data_$Condition), Experiment == "E2")
# This should only include 238_Critical_VAdv1 trials


data_ = data_ %>% filter(!is.na(data_$Condition))

cor.test(data_$LogRT, data_$SurprisalReweighted)


library(ggplot2)
data__ = data_ %>% group_by(Experiment, deletion_rate, predictability_weight, condition, noun, Condition, HasRC, HasSC, compatible, EmbeddingBias) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogRT=mean(LogRT, na.rm=TRUE), rt=mean(rt, na.rm=TRUE))

plot = ggplot(data__ %>% filter(deletion_rate==0.5, predictability_weight==1), aes(x=LogRT, y=SurprisalReweighted, group=Condition, color=paste(HasSC, HasRC))) + geom_point(aes(shape=Experiment)) + theme_classic() + ylab("Surprisal") + theme(legend.position = "none")
plot = ggplot(data__ %>% filter(Experiment %in% c("E1", "E2"), deletion_rate==0.5, predictability_weight==1), aes(x=LogRT, y=SurprisalReweighted)) + geom_smooth(aes(group=Experiment, linetype=Experiment), method="lm", se=F) + geom_point(aes(shape=Experiment, group=Condition, color=paste(HasSC, HasRC)), alpha=0.5) + theme_classic() + ylab("Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/analyze_Model_PlotForExpt12_Joint_ModelHuman_R.pdf", width=3, height=3) # here, full is E1 and dotted is E2

