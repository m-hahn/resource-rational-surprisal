library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
#data1 = read.csv("../previous/study5_replication/Submiterator-master/all_trials.tsv", sep="\t") %>% mutate(distractor=NA, group=NULL)
#data2 = read.csv("../experiment1/Submiterator-master/trials_byWord.tsv", sep="\t") %>% mutate(workerid=workerid+1000)
data3 = read.csv("../experiment2/Submiterator-master/trials-experiment2.tsv", sep="\t") %>% mutate(workerid=workerid+2000)

#data = rbind(data1, data2, data3)
data = rbind(data3)

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

            
#model_E0 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E0.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2)
#model_E1 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E1.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2) %>% filter(!(Item %in% c("Mixed_0", "Mixed_10", "Mixed_20", "ItemMixed_22", "ItemMixed_27", "ItemMixed_28", "ItemMixed_30", "ItemMixed_32")))
model_E2 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t") %>% rename(SurprisalsWithThat=S1, SurprisalsWithoutThat=S2) %>% filter(!grepl("245_", Item))
    
model = rbind(model_E2)   

model$item = model$Item

wordFreqBNC = read.csv("stimuli-bnc-frequencies.tsv", sep="\t") %>% mutate(BNCLogWordFreq = log(Frequency), Word=LowerCaseToken)
model = merge(model, wordFreqBNC, by=c("Word"))

wordFreqCOCA = read.csv("stimuli-coca-frequencies.tsv", sep="\t") %>% mutate(COCALogWordFreq = log(Frequency), Word=LowerCaseToken)
model = merge(model, wordFreqCOCA, by=c("Word"))
# TODO some trials are lost, so some words are missing

#sink("analyze_Model_IncludingNoTP_E2_R_AICs.tsv")
#cat("deletion_rate", "predictability_weight", "AICRaw", "AIC", "AICFull", "N", "None\n", sep="\t")
#sink()

#configs = unique(model %>% select(deletion_rate, predictability_weight))
#for(i in (1:nrow(configs))) {
   delta=0.5 #configs$deletion_rate[[i]]
   lambda=1 #configs$predictability_weight[[i]]
   
   model_ = model %>% filter(Region == "V1_0", deletion_rate == delta, predictability_weight == lambda) %>% group_by(Noun, item, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), BNCLogWordFreq=mean(BNCLogWordFreq, na.rm=TRUE), COCALogWordFreq=mean(COCALogWordFreq, na.rm=TRUE))
   model_$HasSC = !grepl("NoSC", model_$Condition)
   model_$HasRC = grepl("RC", model_$Condition)
   model_$compatible = grepl("_co", model_$Condition)
   
   data_ = merge(data %>% filter(Region == "REGION_3_0"), model_ %>% rename(noun=Noun), by=c("noun", "item", "HasSC", "HasRC", "compatible"), all.x=TRUE)
   
   library(lme4)
   modelSurprisal = (lmer(LogRT ~ SurprisalReweighted + (1|noun) + (1|item) + (1|workerid), data=data_))
 
  sink("output/analyze_Model_IncludingNoTP_E2.R.txt")
  print(summary(modelSurprisal))
  sink()
#   aicTotallyBare = AIC(modelSurprisal)  
#   modelSurprisal = (lmer(LogRT ~ SurprisalReweighted + BNCLogWordFreq + COCALogWordFreq + (1|noun) + (1|item) + (1|workerid), data=data_))
#   aicBare = AIC(modelSurprisal)  
#   modelSurprisal = (lmer(LogRT ~ SurprisalReweighted + BNCLogWordFreq + COCALogWordFreq + EmbeddingBias.C*HasSC.C + EmbeddingBias.C*HasRC.C + EmbeddingBias.C*compatible.C + compatible.C*HasRC.C + (1|noun) + (1|item) + (1|workerid), data=data_))
#   aicFull = AIC(modelSurprisal)  
#   cat(i, delta, lambda, aicTotallyBare, aicBare, aicFull, sum(!is.na(data_$SurprisalReweighted)), "\n", sep="\t")
#   sink("analyze_Model_IncludingNoTP_E2_R_AICs.tsv", append=TRUE)
#   cat(i, delta, lambda, aicTotallyBare, aicBare, aicFull, sum(!is.na(data_$SurprisalReweighted)), "\n", sep="\t")
#   sink()
#}
#  
   #5315.24

