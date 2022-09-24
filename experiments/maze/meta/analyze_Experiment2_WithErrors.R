library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data1 = read.csv("../previous/study5_replication/Submiterator-master/all_trials.tsv", sep="\t") %>% mutate(distractor=NA, group=NULL)
data2 = read.csv("../experiment1/Submiterator-master/trials_byWord.tsv", sep="\t") %>% mutate(workerid=workerid+1000)
data3 = read.csv("../experiment2/Submiterator-master/trials-experiment2.tsv", sep="\t") %>% mutate(workerid=workerid+2000)

data = rbind(data1, data2, data3)

# Exclude participants with excessive error rates
trialsWithError = data %>% group_by(workerid, item) %>% summarise(HasError = max(correct == "no"))
participantsByErrors = trialsWithError %>% group_by(workerid) %>% summarise(Errors = mean(HasError))
plot = ggplot(participantsByErrors, aes(x=Errors)) + geom_histogram() + theme_bw() + xlab("Fraction of Items with Error")

participantsByErrorsBySlide = data %>% filter(correct != "none") %>% group_by(workerid) %>% summarise(ErrorsBySlide = mean(correct == "no"))

plot = ggplot(participantsByErrorsBySlide, aes(x=ErrorsBySlide)) + geom_histogram() + theme_bw() + xlab("Fraction of Words with Error")
ggsave(plot, file="figures/slides-errors.pdf", width=3, height=3)

plot = ggplot(data %>% mutate(condition_group = ifelse(condition == "filler", "Filler", "Critical")) %>% group_by(condition_group, wordInItem) %>% summarise(error_rate=mean(correct == "no")), aes(x=wordInItem, y=error_rate, group=condition_group, color=condition_group)) + geom_line()
plot = ggplot(data %>% group_by(wordInItem) %>% summarise(error_rate=mean(correct == "no")), aes(x=wordInItem, y=error_rate)) + geom_line() + xlab("Word Number") + ylab("Error Rate")
ggsave(plot, file="figures/errors-by-position.pdf", width=5, height=3)

# Only keep critical sentences
data = data %>% filter(condition != "filler", rt > 1)

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

data = data  %>% filter(Region == "REGION_3_0")

data$correct_ = (data$correct == "yes")
data$correct.C = data$correct_ - mean(data$correct_, na.rm=TRUE)

# Mixed-Effects Analysis
library(brms)
model = (brm(LogRT ~ correct.C * EmbeddingBias.C + correct.C * compatible.C + compatible.C * EmbeddingBias.C + HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C + HasSC.C * EmbeddingBias.C + compatible.C * EmbeddingBias.C + (1+compatible.C+HasSC.C+HasRC.C+compatible.C*HasRC.C+correct.C+correct.C*compatible.C|noun) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C + HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C + correct.C + correct.C*compatible.C + correct.C * EmbeddingBias.C|workerid) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C +HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C+ correct.C + correct.C*compatible.C + correct.C * EmbeddingBias.C|item), data=data, cores=4))

sink("analyze_WithErrors.R.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze_WithErrors.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$Depth = samples$b_HasRC.C
samples$EmbeddingBias = samples$b_EmbeddingBias.C
samples$Embedded = samples$b_HasSC.C
samples$Compatible = samples$b_compatible.C
plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Compatible"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/analyze_WithErrors_R_posterior-histograms-main_effects.pdf", width=5, height=5)

samples[["Depth:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:HasRC.C"]]
samples[["Embedded:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:HasSC.C"]]
samples[["Depth:Compatible"]] = samples[["b_compatible.C:HasRC.C"]]
samples[["Compatible:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:compatible.C"]]
plot = mcmc_areas(samples, pars=c("Depth:EmbeddingBias", "Embedded:EmbeddingBias", "Depth:Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/analyze_WithErrors_R_posterior-histograms-interactions.pdf", width=5, height=5)


samples[["Correct"]] = samples[["b_correct.C"]]
samples[["Correct:Compatible"]] = samples[["b_correct.C:compatible.C"]]
samples[["Correct:EmbeddingBias"]] = samples[["b_correct.C:EmbeddingBias.C"]]
plot = mcmc_areas(samples, pars=c("Correct", "Correct:Compatible", "Correct:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/analyze_WithErrors_R_posterior-histograms-interactionsWithCorrect.pdf", width=5, height=4)



