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


# Mixed-Effects Analysis
library(brms)
model = (brm(LogRT ~ HasRC.C * compatible.C + HasRC.C + HasSC.C + compatible.C + (1+compatible.C+HasSC.C+HasRC.C|noun) + (1+compatible.C + HasSC.C + HasSC.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C|workerid) + (1+compatible.C + compatible.C + HasSC.C +HasSC.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C|item), data=data %>% filter(Region == "REGION_3_0"), cores=4, iter=6000))

samples = posterior_samples(model)

perNounEffects = data.frame()

for(noun_ in unique(data$noun)) {
  if(!is.na(noun_)) {
    WP_EmbeddingBias = (data %>% filter(noun == noun_))$EmbeddingBias.C[[1]]
    effectForNoun = samples[[paste("r_noun[", noun_, ",Intercept]", sep="")]] + max(data$HasSC.C, na.rm=TRUE) * samples[[paste("r_noun[", noun_, ",HasSC.C]", sep="")]]
    perNounEffectHere = data.frame(estimate=c(mean(effectForNoun)), lower=c(quantile(effectForNoun, 0.025)), upper=c(quantile(effectForNoun, 0.975)), embBias=c(WP_EmbeddingBias), noun=noun_)
    perNounEffects = rbind(perNounEffects, perNounEffectHere)
  }
}

sink("output/extractPerNounIntercepts_Raw_R.txt")
cor.test(perNounEffects$embBias, perNounEffects$estimate)
sink()

write.table(perNounEffects, file="output/extractPerNounIntercepts_Raw.R.tsv", sep="\t")

# get a posterior for the correlation

correlations = c()
for(i in 120*(1:100)) {
  perNounEffects = data.frame()
  
  for(noun_ in unique(data$noun)) {
    if(!is.na(noun_)) {
      WP_EmbeddingBias = (data %>% filter(noun == noun_))$EmbeddingBias.C[[1]]
      effectForNoun = samples[[paste("r_noun[", noun_, ",Intercept]", sep="")]] + max(data$HasSC.C, na.rm=TRUE) * samples[[paste("r_noun[", noun_, ",HasSC.C]", sep="")]]
      perNounEffectHere = data.frame(estimate=c(effectForNoun[[i]]), embBias=c(WP_EmbeddingBias), noun=noun_)
      perNounEffects = rbind(perNounEffects, perNounEffectHere)
    }
  }
  correlation = cor(perNounEffects$embBias, perNounEffects$estimate)
  cat(i, correlation, "\n")
  correlations = c(correlations, correlation)
}

sink("output/extractPerNounIntercepts_Raw_R_posterior.txt")
cat(mean(correlations), " ", quantile(correlations, 0.025), quantile(correlations, 0.975))
sink()



