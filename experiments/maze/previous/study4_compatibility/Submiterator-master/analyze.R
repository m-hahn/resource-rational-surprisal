library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data = read.csv("all_trials.tsv", sep="\t")

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

data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)

# Code Embedding Bias
data = data %>% mutate(EmbeddingBias = True_False_False-False_False_False)
data = data %>% mutate(EmbeddingBias.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))

# Code Conditions
data = data %>% mutate(compatible = ifelse(condition %in% c("critical_SCRC_compatible", "critical_compatible"), TRUE, FALSE))
data = data %>% mutate(HasSC = ifelse(condition == "critical_NoSC", FALSE, TRUE))

# Contrast Coding
data$HasSC.C = data$HasSC - 0.8
data$compatible.C = ifelse(!data$HasSC, 0, ifelse(data$compatible, 0.5, -0.5))
# Log-Transform Reading Times
data$LogRT = log(data$rt)

# Center trial order
data$trial = data$trial - mean(data$trial, na.rm=TRUE)


# Mixed-Effects Analysis
library(brms)
model = (brm(LogRT ~ compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + (1+compatible.C|noun) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + EmbeddingBias.C |workerid) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + EmbeddingBias.C|item), data=data %>% filter(Region == "REGION_3_0")))

sink("output/analyze.R.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$EmbeddingBias = samples$b_EmbeddingBias.C
samples$Embedded = 0
samples$Compatible = samples$b_compatible.C
plot = mcmc_areas(samples, pars=c("EmbeddingBias", "Compatible"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects.pdf", width=5, height=2)

samples[["Compatible:EmbeddingBias"]] = samples[["b_compatible.C:EmbeddingBias.C"]]
plot = mcmc_areas(samples, pars=c("Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-interactions.pdf", width=5, height=1)


plot = mcmc_areas(samples, pars=c("EmbeddingBias", "Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms.pdf", width=5, height=3)


embeddingBiasSamples = data.frame(EmbeddingBiasWithinTwo = samples$b_EmbeddingBias.C)
plot = mcmc_areas(embeddingBiasSamples, prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-EmbeddingBias.pdf", width=5, height=5)


sink("output/analyze.R_posteriors.txt")
cat("b_compatible.C ", mean(samples$b_compatible.C<0), "\n")
cat("EmbeddingBiasAcrossTwoThree ", mean(embeddingBiasSamples$EmbeddingBiasAcrossTwoThree>0), "\n")

RTCompatible = exp(samples$b_Intercept + max(data$compatible.C) * samples[["Compatible"]])
RTIncompatible = exp(samples$b_Intercept + min(data$compatible.C) * samples[["Compatible"]])
CompatibilityEffect = RTCompatible-RTIncompatible
cat("Effect Compatibility within Two", mean(CompatibilityEffect), " ", quantile(CompatibilityEffect, 0.025), " ", quantile(CompatibilityEffect, 0.975), "\n")




# Effect of Embedding Bias in Raw RTs
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]

# in Embedded condition (Two/Three)
RTReportTwo = exp(samples$b_Intercept + samples[["r_noun[report,Intercept]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + 0 * 0))
RTFactTwo = exp(samples$b_Intercept + 0 * 0 + samples[["r_noun[fact,Intercept]"]] + 0 * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + 0 * 0))
FactReportDifferenceTwo = RTFactTwo - RTReportTwo 
samples$FactReportDifferenceTwo = FactReportDifferenceTwo
cat("Fact/Report Difference in Two", mean(FactReportDifferenceTwo), " ", quantile(FactReportDifferenceTwo, 0.025), " ", quantile(FactReportDifferenceTwo, 0.975), "\n")

# in One condition
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]
RTReportOne = exp(samples$b_Intercept + 0 * 0 + samples[["r_noun[report,Intercept]"]] + 0 * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + 0 * 0))
RTFactOne = exp(samples$b_Intercept + 0 * 0 + samples[["r_noun[fact,Intercept]"]] + 0 * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + 0 * 0))
FactReportDifferenceOne = RTFactOne - RTReportOne
samples$FactReportDifferenceOne = FactReportDifferenceOne
cat("Fact/Report Difference within One", mean(FactReportDifferenceOne), " ", quantile(FactReportDifferenceOne, 0.025), " ", quantile(FactReportDifferenceOne, 0.975), "\n")


sink()
#samples$CompatibilityEffect_Two = CompatibilityEffect

plot = mcmc_areas(samples, pars=c("FactReportDifferenceTwo", "CompatibilityEffect_Two"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-RawEffects.pdf", width=5, height=2)




#plot = mcmc_areas(samples, pars=c("DepthEffect", "FactReportDifference"), prob=.95, n_dens=32, adjust=5)
#ggsave(plot, file="figures/posterior-histograms-RawRTs.pdf", width=5, height=4)

# In this experiment, only Two condition
samples$b_HasSC.C = 0

EmbeddingBias.C = c(EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact) #, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact)
#HasSC = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)-0.5
#HasRC = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)-0.5
HasSC.C = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)-0.8
HasRC.C = c(0.5, 0.5, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE)-0.5
HasSC = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
HasRC = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE)
compatible.C = c(0.5, 0.5, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE)-0.5
compatible = c(FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE)
#Compatible = c(0.5, 0.5, 0, 0, 1, 1, 0, 0, 1, 1)-0.5
RTPred = exp(mean(samples$b_Intercept) + HasSC.C * mean(samples$b_HasSC.C) + EmbeddingBias.C * mean(samples$b_EmbeddingBias.C) + HasSC.C * EmbeddingBias.C * mean(0) + compatible.C * mean(samples$b_compatible.C) + compatible.C*EmbeddingBias.C*mean(samples[["b_compatible.C:EmbeddingBias.C"]]))

predictions = data.frame(EmbeddingBias.C=EmbeddingBias.C, HasSC.C=HasSC.C, HasRC.C=HasRC.C, RTPred=RTPred, HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias.C-mean(data$EmbeddingBias.C-data$EmbeddingBias), compatible=compatible, compatible.C=compatible.C)
library(ggplot2)
dataPlot = ggplot(predictions, aes(x=EmbeddingBias.C, y=RTPred, group=paste(HasSC.C, HasRC.C), color=paste(HasSC.C, HasRC.C))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC.C, HasRC.C, noun, EmbeddingBias.C) %>% summarise(rt=mean(rt)), aes(x=EmbeddingBias.C, y=rt))

data$HasRC.C = -0.5
data$HasRC = FALSE

data$HasSC.C = 0.5
data$HasSC = TRUE

predictions$condition = paste(predictions$HasSC, predictions$HasRC, predictions$compatible)
dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=log(RTPred), group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC))) + geom_smooth(se=F, method="lm", aes(linetype=compatible)) + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC), linetype=compatible)) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-fit.pdf", height=3.5, width=1.8)


dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=(RTPred), group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC))) + geom_smooth(se=F, method="lm", aes(linetype=compatible)) + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=exp(rt), group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC), linetype=compatible)) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-fit_NoLogTransform.pdf", height=3.5, width=1.8)




dataPlot = ggplot(data %>% filter(Region == "REGION_3_0"), aes(x=EmbeddingBias, y=LogRT, group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC))) + geom_smooth(se=F, method="lm", aes(linetype=compatible)) + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC), linetype=compatible)) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-raw.pdf", height=3.5, width=1.8)



#dataPlot = ggplot(data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) +  geom_point()

#+ scale_color_manual(values = c("FALSE_FALSE" = "#F8766D", "TRUE_FALSE"="#00BA38", "TRUE_TRUE"="#619CFF"))


