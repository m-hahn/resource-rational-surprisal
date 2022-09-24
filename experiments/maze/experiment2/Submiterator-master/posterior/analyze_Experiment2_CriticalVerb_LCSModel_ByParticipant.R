library(ggplot2)
library(dplyr)
library(tidyr)


modelSurprisal = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t") %>% filter(Region == "V1_0")

#modelSurprisal = merge(modelSurprisal, nounFreqs %>% rename(Noun=noun), by="Noun")
#modelSurprisal = modelSurprisal%>%mutate(True_Minus_False = True_False_False-False_False_False)  

#quit()

modelsWithEnoughObservations = unique(modelSurprisal %>% select(ID, Noun)) %>% group_by(ID) %>% summarise(observations = NROW(Noun)) %>% filter(observations > 50)

modelSurprisal = merge(modelSurprisal, modelsWithEnoughObservations, by=c("ID"))

modelSurprisalSmoothed = data.frame()

configs = unique(modelSurprisal %>% select(deletion_rate, predictability_weight))
for(i in (1:nrow(configs))) {
   delta = configs$deletion_rate[[i]]
   lambda = configs$predictability_weight[[i]]
   relevant = modelSurprisal %>% filter(abs(deletion_rate-delta) <= 0.0, abs(predictability_weight-lambda) <= 0.0)
   relevant = relevant %>% group_by(Noun, Item, Region, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction))
   relevant$deletion_rate = delta
   relevant$predictability_weight = lambda
   modelSurprisalSmoothed = rbind(modelSurprisalSmoothed, as.data.frame(relevant))
}

modelSurprisalSmoothed = as.data.frame(modelSurprisalSmoothed)

# modelSurprisalSmoothed = merge(modelSurprisalSmoothed, nounFreqs %>% rename(Noun=noun), by="Noun")
# relevant = modelSurprisalSmoothed %>% filter(deletion_rate==0.5, predictability_weight==1.0) %>% mutate(True_Minus_False=True_False_False-False_False_False)
# relevant %>% group_by(True_Minus_False, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% group_by(Condition) %>% summarise(corr=cor(True_Minus_False, SurprisalReweighted))


# human %>% group_by(True_Minus_False, condition) %>% summarise(SurprisalReweighted=mean(Surprisal_0.5_1)) %>% group_by(condition) %>% summarise(corr=cor(True_Minus_False, SurprisalReweighted))

# option 1: per model run

# option 2: smooth model predictions over adjacent cells

data = read.csv("../trials-experiment2.tsv",sep="\t") %>% mutate(group=238)


#> unique(data %>% select(group, workerid)) %>% group_by(group) %>% summarise(N=NROW(workerid))

# Exclusion
participantsByErrorsBySlide = data %>% filter(correct != "none") %>% group_by(workerid) %>% summarise(ErrorsBySlide = mean(correct == "no"))
data = merge(data, participantsByErrorsBySlide, by=c("workerid"))
data = data %>% filter(ErrorsBySlide < 0.2)



data = data %>% filter(condition != "filler", rt > 1)

nounFreqs = read.csv("../../../../../materials/nouns/corpus_counts/wikipedia/results/results_counts4NEW.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)

nounFreqs2 = read.csv("../../../../../materials/nouns/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]



data$Noun = data$noun
#data = merge(data, model, by=c("Noun"), all.x=TRUE)

data$noun = data$Noun
data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)



data = data %>% mutate(True_True_False.C = True_True_False-mean(True_True_False, na.rm=TRUE))
data = data %>% mutate(True_False_False.C = True_False_False-mean(True_False_False, na.rm=TRUE))
#data = data %>% mutate(grammatical = (condition == "grammatical"))
#data = data %>% mutate(grammatical.C = grammatical-mean(grammatical, na.rm=TRUE))
data = data %>% mutate(True_Minus_False.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))
data = data %>% mutate(False_False_False.C = False_False_False-mean(False_False_False, na.rm=TRUE))
data = data %>% mutate(True_Minus_False = True_False_False-False_False_False)




#data$Stimuli238 = grepl("238_", data$item)



data = data %>% mutate(compatible = ifelse(condition %in% c("critical_SCRC_compatible", "critical_compatible"), TRUE, FALSE))

data = data %>% mutate(HasSC = ifelse(condition == "critical_NoSC", FALSE, TRUE))
data = data %>% mutate(HasRC = ifelse(condition %in% c("critical_SCRC_compatible", "critical_SCRC_incompatible"), TRUE, FALSE))


data$HasRC.C = resid(lm(HasRC ~ HasSC, data=data))




#data = data %>% mutate(compatible = ifelse(workerid %in% invertedParticipants, !compatible, compatible))

#data = data %>% mutate(HasSC = ifelse(condition == "critical_NoSC", FALSE, TRUE))

data$HasSC.C = data$HasSC - mean(data$HasSC, na.rm=TRUE)
data$compatible.C = data$compatible - mean(data[data$HasSC,]$compatible, na.rm=TRUE)
data[!data$HasSC,]$compatible.C = 0

library(lme4)
#summary(glmer(correct ~ compatible.C * True_Minus_False.C + (1|Noun) + (1|workerid) + (1|Continuation), data=data %>% filter(Region == "REGION_3_0"), family="binomial"))


data = data %>% filter(rt > 0, correct == "yes")

data = data %>% filter(rt < quantile(data$rt, 0.99))


#data = data %>% filter(rt > quantile(data$rt, 0.01)) # post-hoc robustness check -- also report this version in the SI

data = data %>% mutate(condition.C = ifelse(condition == "condition_1", -0.5, 0.5))
#data = data %>% mutate(SurpDiff.C = SurpDiff-mean(SurpDiff, na.rm=TRUE))
#data = data %>% mutate(Surprisal.C = Surprisal-mean(Surprisal, na.rm=TRUE))

data$LogRT = log(data$rt)

#critical[0].push({s: "The belief of the principal has been mentioned in the newspaper.", a : "x-x-x become see map elsewhere ago code personnel bed eat proceeded."   })
#critical[0].push({s: "The belief of the principals has been mentioned in the newspaper.", a : "x-x-x indeed say ran formatting gas poet procedure thy got happiness."  })
#critical[0].push({s: "The belief of the principal have been mentioned in the newspaper.", a : "x-x-x remove lot sum including milk debt equipment son off occurring."  })
#critical[0].push({s: "The belief of the principals have been mentioned in the newspaper.", a : "x-x-x myself fat ran irradiated wait talk operation oil am specified."   })




library(lme4)
library(ggplot2)

data$trial = data$trial - mean(data$trial, na.rm=TRUE)


###################################################################
library(brms)

#modelSurprisalSmoothed

#library(rstan)




#   sink("output/analyze_Experiment2_CriticalVerb_LCSModel_ByParticipant.R.tsv", append=FALSE)
#   cat("deletion_rate", "predictability_weight", "AIC", "nrow", "\n", sep="\t")
#   sink()
configs = unique(modelSurprisal %>% select(deletion_rate, predictability_weight))

overall_data = data.frame()

human = data %>% filter(Region == "REGION_3_0") %>% select(noun, item, condition, workerid, rt, LogRT, True_Minus_False)

human$item = as.character(human$item)
human$item = ifelse(human$item == "238_Critical_VAdv1", "238_Critical_Vadv1", human$item)
modelSurprisalSmoothed$Item = as.character(modelSurprisalSmoothed$Item)
modelSurprisalSmoothed$Item = ifelse(modelSurprisalSmoothed$Item == "238_Critical_VAdv1", "238_Critical_Vadv1", modelSurprisalSmoothed$Item)

for(i in (1:nrow(configs))) {
   delta = configs$deletion_rate[[i]];    lambda = configs$predictability_weight[[i]] ;    relevant = modelSurprisalSmoothed %>% filter((deletion_rate == delta), (predictability_weight == lambda)) %>% filter(Region == "V1_0") %>% rename(noun=Noun, item=Item) %>% mutate(condition = case_when(Condition == "NoSC_ne" ~ "critical_NoSC", Condition == "SC_co" ~ "critical_compatible", Condition == "SC_in" ~ "critical_incompatible", Condition == "SCRC_co" ~ "critical_SCRC_compatible", Condition == "SCRC_in" ~ "critical_SCRC_incompatible", FALSE ~ "other")) %>% group_by(noun, item, condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
   relevant[[paste("Surprisal_", delta, "_", lambda, sep="")]] = relevant$SurprisalReweighted
   relevant$SurprisalReweighted = NULL
   human = merge(relevant, human, by=c("noun", "item", "condition"), all.y=TRUE)
   if(any(is.na(human[[paste("Surprisal_", delta, "_", lambda, sep="")]]))) {
      cat("ERROR: MISSING DATA", delta, lambda, "\n")
      human[[paste("Surprisal_", delta, "_", lambda, sep="")]] = NULL
   } else {
      cat("DONE", delta, lambda, nrow(human), "\n")
   }

}

noun = human$noun
item = human$item
condition = human$condition
rt = human$rt
LogRT = human$LogRT

workerid = as.numeric(as.factor(human$workerid))
item = as.numeric(as.factor(human$item))
#condition = as.numeric(as.factor(human$condition))
noun = as.numeric(as.factor(human$noun))
True_Minus_False = human$True_Minus_False

model_predictions = human
model_predictions$noun = NULL ; model_predictions$item = NULL; model_predictions$condition = NULL; model_predictions$rt = NULL; model_predictions$LogRT = NULL; model_predictions$workerid=NULL ; model_predictions$True_Minus_False=NULL

write.table(LogRT, file="forStan/LogRT.tsv", quote=F, sep="\t")
write.table(model_predictions, file="forStan/predictions.tsv", quote=F, sep="\t")
write.table(workerid, file="forStan/subjects.tsv", quote=F, sep="\t")
write.table(item, file="forStan/items.tsv", quote=F, sep="\t")
write.table(condition, file="forStan/conditions.tsv", quote=F, sep="\t")
write.table(noun, file="forStan/nouns.tsv", quote=F, sep="\t")
write.table(True_Minus_False, file="forStan/embeddingBias.tsv", quote=F, sep="\t")

#fit1 <- stan(  file = "stanmodel1.stan",   data = schools_data,       chains = 4,  warmup = 1000,      iter = 2000,        cores = 4,   refresh = 0  )




#write.table(overall_data, file="output/analyze_Experiment2_CriticalVerb_LCSModel_ByParticipant.R.tsv")


