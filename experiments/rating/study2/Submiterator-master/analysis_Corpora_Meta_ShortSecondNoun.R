library(dplyr)
library(tidyr)
library(lme4)


data = read.csv("all_trials.tsv", sep="\t")

########
responses = data %>% mutate(correct = (data$given_answer == data$correct_answer)) %>% group_by(workerid) %>% summarise(correct = mean(correct, na.rm=TRUE))
cat("Fraction of excluded participants: ", mean(responses$correct < 0.8), "\n")
data = merge(data, responses, by=c("workerid"))
data = data %>% filter(correct >= 0.8)
data = data %>% filter(condition!="filler")

#######################################
# Identify nouns and frames
noun = c()
remainder = c()
sentences = as.character(data$sentence)
for(i in (1:nrow(data))) {
        strin = strsplit(sentences[i], " ")
        noun = c(noun, strin[[1]][2])
        remainder = c(remainder, paste(strin[[1]][5], strin[[1]][6], sep="_"))
}
data$noun = noun
data$remainder = remainder

# PROBLEM scores absent for confirmation, revelation


###################

nounFreqs = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)

nounFreqs2 = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)

#######################

data = data %>% mutate(True_True_False.C = True_True_False-mean(True_True_False, na.rm=TRUE))
data = data %>% mutate(True_False_False.C = True_False_False-mean(True_False_False, na.rm=TRUE))
data = data %>% mutate(Grammatical = (condition == "0"))
data = data %>% mutate(Grammatical.C = Grammatical-mean(Grammatical, na.rm=TRUE))
data = data %>% mutate(EmbeddingBias.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))
data = data %>% mutate(False_False_False.C = False_False_False-mean(False_False_False, na.rm=TRUE))
data = data %>% mutate(EmbeddingBias = True_False_False-False_False_False)

############################

library(brms)

modelOrdinal = (brm(rating ~ Grammatical.C*EmbeddingBias.C + (1+Grammatical.C+EmbeddingBias.C+Grammatical.C*EmbeddingBias.C|workerid) + (1+Grammatical.C|noun) + (1+Grammatical.C+EmbeddingBias.C+Grammatical.C*EmbeddingBias.C|remainder), data=data, family="cumulative"))

samples = posterior_samples(modelOrdinal)
print(summary(modelOrdinal))
print("Posterior of opposite sign for interaction:", mean(posterior_samples(modelOrdinal)[["b_Grammatical.C:EmbeddingBias.C"]] > 0))

library(bayesplot)
library(ggplot2)
plot = mcmc_areas(samples, pars=c("b_Grammatical.C", "b_EmbeddingBias.C", "b_Grammatical.C:EmbeddingBias.C"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects.pdf", width=5, height=4.5)

plot = mcmc_areas(samples, pars=c("b_Grammatical.C:EmbeddingBias.C"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-interactions.pdf", width=5, height=1.5)


plot = ggplot(data %>% group_by(noun, EmbeddingBias, Grammatical) %>% summarise(rating=mean(rating)), aes(x=EmbeddingBias, y=rating, group=Grammatical, color=Grammatical)) + geom_smooth(method="lm") + geom_label(aes(label=noun)) + xlab("Embedding Bias") + ylab("Difficulty Rating") + theme_bw()
ggsave(plot, file="figures/rating_understand-logodds-byNoun-LogRatio.pdf", height=6, width=6)





sink("output/posterior.txt")
print(summary(modelOrdinal))
cat("\n")
cat("Posterior:")
cat(mean(samples[["b_grammatical.C:True_Minus_False.C"]] >0))
sink()

