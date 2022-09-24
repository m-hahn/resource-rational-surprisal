library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data = read.csv("all_trials.tsv",sep="\t")

# Recruitment overshot by 4 subjects
data = data %>% filter(workerid <= 30)

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
nounFreqs = read.csv("../../../../../materials/nouns/corpus_counts/wikipedia/results/results_counts4NEW.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)

nounFreqs2 = read.csv("../../../../../materials/nouns/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data$noun = data$Noun
data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)

# Code Embedding Bias
data = data %>% mutate(EmbeddingBias = True_False_False-False_False_False)
data = data %>% mutate(EmbeddingBias.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))

# Code Conditions
data = data %>% mutate(HasSC = ifelse(condition == "condition_0", FALSE, TRUE))

# Contrast Coding
data$HasSC.C = data$HasSC - 0.5

# Log-Transform Reading Times
data$LogRT = log(data$rt)

# Center trial order
data$trial = data$trial - mean(data$trial, na.rm=TRUE)


# Mixed-Effects Analysis
library(brms)
model = (brm(LogRT ~ EmbeddingBias.C + HasSC.C * EmbeddingBias.C + EmbeddingBias.C + (1+HasSC.C|noun) + (1 + EmbeddingBias.C + HasSC.C + HasSC.C * EmbeddingBias.C |workerid) + (1+EmbeddingBias.C + HasSC.C +HasSC.C * EmbeddingBias.C + EmbeddingBias.C|item), data=data %>% filter(Region == "REGION_3_0")))

sink("output/analyze.R.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$EmbeddingBias = samples$b_EmbeddingBias.C
samples$Embedded = samples$b_HasSC.C
plot = mcmc_areas(samples, pars=c("EmbeddingBias", "Embedded"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects.pdf", width=5, height=5)

samples[["Embedded:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:HasSC.C"]]
plot = mcmc_areas(samples, pars=c("Embedded:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-interactions.pdf", width=5, height=5)


plot = mcmc_areas(samples, pars=c("EmbeddingBias", "Embedded", "Embedded:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms.pdf", width=5, height=3)


embeddingBiasSamples = data.frame(EmbeddingBiasWithinOne = samples$b_EmbeddingBias.C + min(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasWithinTwo = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]])
plot = mcmc_areas(embeddingBiasSamples, prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-EmbeddingBias.pdf", width=5, height=2)


sink("output/analyze.R_posteriors.txt")
cat("b_EmbeddingBias.C:HasSC.C ", mean(samples[["b_EmbeddingBias.C:HasSC.C"]]>0), "\n")
cat("EmbeddingBiasAcrossTwoThree ", mean(embeddingBiasSamples$EmbeddingBiasAcrossTwoThree>0), "\n")


# Effect of Embedding Bias in Raw RTs
EmbeddingBiasC_admission = (data %>% filter(noun == "admission"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]

# in Embedded condition (Two/Three)
RTAdmissionTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[admission,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[admission,HasSC.C]"]] + EmbeddingBiasC_admission * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
FactAdmissionDifferenceTwo = RTFactTwo - RTAdmissionTwo 
samples$FactAdmissionDifferenceTwo = FactAdmissionDifferenceTwo
cat("Fact/Admission Difference in Two", mean(FactAdmissionDifferenceTwo), " ", quantile(FactAdmissionDifferenceTwo, 0.025), " ", quantile(FactAdmissionDifferenceTwo, 0.975), "\n")

# in One condition
EmbeddingBiasC_admission = (data %>% filter(noun == "admission"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]
RTAdmissionOne = exp(samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[admission,Intercept]"]] + min(data$HasSC.C) * samples[["r_noun[admission,HasSC.C]"]] + EmbeddingBiasC_admission * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactOne = exp(samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + min(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
FactAdmissionDifferenceOne = RTFactOne - RTAdmissionOne
samples$FactAdmissionDifferenceOne = FactAdmissionDifferenceOne
cat("Fact/Admission Difference within One", mean(FactAdmissionDifferenceOne), " ", quantile(FactAdmissionDifferenceOne, 0.025), " ", quantile(FactAdmissionDifferenceOne, 0.975), "\n")


sink()

plot = mcmc_areas(samples, pars=c("FactAdmissionDifferenceOne", "FactAdmissionDifferenceTwo"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-RawEffects.pdf", width=5, height=2)


#plot = mcmc_areas(samples, pars=c("DepthEffect", "FactAdmissionDifference"), prob=.95, n_dens=32, adjust=5)
#ggsave(plot, file="figures/posterior-histograms-RawRTs.pdf", width=5, height=4)


EmbeddingBias.C = c(EmbeddingBiasC_admission, EmbeddingBiasC_fact, EmbeddingBiasC_admission, EmbeddingBiasC_fact, EmbeddingBiasC_admission, EmbeddingBiasC_fact) #, EmbeddingBiasC_admission, EmbeddingBiasC_fact, EmbeddingBiasC_admission, EmbeddingBiasC_fact)
#HasSC = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)-0.5
#HasRC = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)-0.5
HasSC.C = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)-0.8
HasRC.C = c(0.5, 0.5, FALSE, FALSE, FALSE, FALSE)-0.5
HasSC = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)
HasRC = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE)
#Compatible = c(0.5, 0.5, 0, 0, 1, 1, 0, 0, 1, 1)-0.5
RTPred = exp(mean(samples$b_Intercept) + HasSC.C * mean(samples$b_HasSC.C) + EmbeddingBias.C * mean(samples$b_EmbeddingBias.C) + HasSC.C * EmbeddingBias.C * mean(samples[["Embedded:EmbeddingBias"]]))

data$HasRC.C = 0.5
data$HasRC = FALSE
predictions = data.frame(EmbeddingBias.C=EmbeddingBias.C, HasSC.C=HasSC.C, HasRC.C=HasRC.C, RTPred=RTPred, HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias.C-mean(data$EmbeddingBias.C-data$EmbeddingBias))
library(ggplot2)
dataPlot = ggplot(predictions, aes(x=EmbeddingBias.C, y=RTPred, group=paste(HasSC.C, HasRC.C), color=paste(HasSC.C, HasRC.C))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC.C, HasRC.C, noun, EmbeddingBias.C) %>% summarise(rt=mean(rt)), aes(x=EmbeddingBias.C, y=rt))

dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=log(RTPred), group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC), color=paste(HasSC, HasRC)), alpha=0.5) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-fit.pdf", height=2.5, width=2.5)

dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=(RTPred), group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=exp(rt), group=paste(HasSC, HasRC), color=paste(HasSC, HasRC)), alpha=0.5) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-fit_NoLogTransform.pdf", height=2.5, width=2.5)



