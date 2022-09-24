library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data = read.csv("trials-experiment2.tsv", sep="\t")

# The first batch overshot by 17 subjects.
# Here, excluding those for consistency with planned number of subjects as described in main paper.
# (Note: After error-based exclusion, the number of subjects in the analysis drops from 200 to 186.)
# Including the 17 subjects wouldn't make a difference to the pattern of results.
# In any case, all 217 subjects are included in the meta-analysis (SI Appendix, Section 6.4).
data = data %>% filter(workerid < 100 | workerid > 117)

# Exclude participants with excessive error rates
participantsByErrorsBySlide = data %>% filter(correct != "none") %>% group_by(workerid) %>% summarise(ErrorsBySlide = mean(correct == "no"))
data = merge(data, participantsByErrorsBySlide, by=c("workerid"))
data = data %>% filter(ErrorsBySlide < 0.2)

# Only consider critical trials
data = data %>% filter(condition != "filler")

# Remove trials with incorrect responses
data = data %>% filter(rt > 0, correct == "yes")

# Remove extremely low or extremely high reading times
data = data %>% filter(rt < 10000) #quantile(data$rt, 0.999))
data = data %>% filter(rt > 200) #quantile(data$rt, 0.001))


# Load corpus counts (Wikipedia)
nounFreqs = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/results_counts4NEW.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)

nounFreqs2 = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

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
model = (brm(LogRT ~ HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C + HasSC.C * EmbeddingBias.C + compatible.C * EmbeddingBias.C + (1+compatible.C+HasSC.C+HasRC.C+HasRC.C*compatible.C|noun) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C + HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C|workerid) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C +HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C|item), data=data %>% filter(Region == "REGION_3_0"), cores=4, iter=8000))

sink("output/analyze.R.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$Depth = samples$b_HasRC.C
samples$EmbeddingBias = samples$b_EmbeddingBias.C
samples$Embedded = samples$b_HasSC.C
samples$Compatible = samples$b_compatible.C
plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Compatible"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects_Outliers.pdf", width=5, height=5)

samples[["Depth:EmbeddingBias"]] = samples[["b_HasRC.C:EmbeddingBias.C"]]
samples[["Embedded:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:HasSC.C"]]
samples[["Depth:Compatible"]] = samples[["b_HasRC.C:compatible.C"]]
samples[["Compatible:EmbeddingBias"]] = samples[["b_compatible.C:EmbeddingBias.C"]]
plot = mcmc_areas(samples, pars=c("Depth:EmbeddingBias", "Embedded:EmbeddingBias", "Depth:Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-interactions_Outliers.pdf", width=5, height=5)


plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Compatible", "Depth:EmbeddingBias", "Embedded:EmbeddingBias", "Depth:Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms_Outliers.pdf", width=5, height=4)


embeddingBiasSamples = data.frame(EmbeddingBiasWithinOne = samples$b_EmbeddingBias.C + min(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasAcrossTwoThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasWithinTwo = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + -5.630822e-01 * samples[["b_HasRC.C:EmbeddingBias.C"]], EmbeddingBiasWithinThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + 4.369178e-01 * samples[["b_HasRC.C:EmbeddingBias.C"]])
plot = mcmc_areas(embeddingBiasSamples, prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-EmbeddingBias_Outliers.pdf", width=5, height=5)


sink("output/analyze.R_posteriors.txt")
cat("b_HasRC.C ", mean(samples$b_HasRC.C<0), "\n")
cat("b_compatible.C ", mean(samples$b_compatible.C<0), "\n")
cat("b_EmbeddingBias.C:HasSC.C ", mean(samples[["b_EmbeddingBias.C:HasSC.C"]]>0), "\n")
cat("b_EmbeddingBias.C:HasRC.C ", mean(samples[["b_HasRC.C:EmbeddingBias.C"]]>0), "\n")
cat("EmbeddingBiasAcrossTwoThree ", mean(embeddingBiasSamples$EmbeddingBiasAcrossTwoThree>0), "\n")
cat("EmbeddingBiasWithinTwo ", mean(embeddingBiasSamples$EmbeddingBiasWithinTwo>0), "\n")
cat("EmbeddingBiasWithinThree ", mean(embeddingBiasSamples$EmbeddingBiasWithinThree>0), "\n")

RTThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + max(data$HasRC.C) * samples[["Depth"]])
RTTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + min(data$HasRC.C) * samples[["Depth"]])
DepthEffect = RTThree-RTTwo
samples$DepthEffect = DepthEffect
cat("Effect Depth", mean(DepthEffect), " ", quantile(DepthEffect, 0.025), " ", quantile(DepthEffect, 0.975), "\n")


RTCompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + max(data$compatible.C) * samples[["Compatible"]])
RTIncompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + min(data$compatible.C) * samples[["Compatible"]])
CompatibilityEffect = RTCompatible-RTIncompatible
cat("Effect Compatibility", mean(CompatibilityEffect), " ", quantile(CompatibilityEffect, 0.025), " ", quantile(CompatibilityEffect, 0.975), " ", mean(CompatibilityEffect<0),"\n")


RTCompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + max(data$compatible.C) * samples[["Compatible"]] + min(data$HasRC.C) * samples[["Depth"]] + max(data$compatible.C) * min(data$HasRC.C) * samples[["Depth:Compatible"]])
RTIncompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + min(data$compatible.C) * samples[["Compatible"]] + min(data$HasRC.C) * samples[["Depth"]] + min(data$compatible.C) * min(data$HasRC.C) * samples[["Depth:Compatible"]])
CompatibilityEffect = RTCompatible-RTIncompatible
samples$CompatibilityEffect_Two = CompatibilityEffect
cat("Effect Compatibility within Two", mean(CompatibilityEffect), " ", quantile(CompatibilityEffect, 0.025), " ", quantile(CompatibilityEffect, 0.975), " ", mean(CompatibilityEffect<0), "\n")


RTCompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + max(data$compatible.C) * samples[["Compatible"]] + max(data$HasRC.C) * samples[["Depth"]] + max(data$compatible.C) * max(data$HasRC.C) * samples[["Depth:Compatible"]])
RTIncompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + min(data$compatible.C) * samples[["Compatible"]] + max(data$HasRC.C) * samples[["Depth"]] + min(data$compatible.C) * max(data$HasRC.C) * samples[["Depth:Compatible"]])
CompatibilityEffect = RTCompatible-RTIncompatible
samples$CompatibilityEffect_Three = CompatibilityEffect
cat("Effect Compatibility within Three", mean(CompatibilityEffect), " ", quantile(CompatibilityEffect, 0.025), " ", quantile(CompatibilityEffect, 0.975), " ", mean(CompatibilityEffect<0),"\n")





# Effect of Embedding Bias in Raw RTs
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]


# in Embedded condition (Two/Three)
RTReportEmbedded = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactEmbedded = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
FactReportDifferenceEmbedded = RTFactEmbedded - RTReportEmbedded 
samples$FactReportDifferenceEmbedded = FactReportDifferenceEmbedded
cat("Fact/Report Difference across Two/Three", mean(FactReportDifferenceEmbedded), " ", quantile(FactReportDifferenceEmbedded, 0.025), " ", quantile(FactReportDifferenceEmbedded, 0.975), "\n")

RTReportTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + min(data$HasRC.C) * (samples[["Depth"]] + samples[["r_noun[report,HasRC.C]"]]))
RTFactTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + min(data$HasRC.C) * (samples[["Depth"]] + samples[["r_noun[fact,HasRC.C]"]]))
FactReportDifferenceTwo = RTFactTwo - RTReportTwo 
samples$FactReportDifferenceTwo = FactReportDifferenceTwo
cat("Fact/Report Difference in Two", mean(FactReportDifferenceTwo), " ", quantile(FactReportDifferenceTwo, 0.025), " ", quantile(FactReportDifferenceTwo, 0.975), "\n")

RTReportThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + max(data$HasRC.C) * samples[["Depth"]] + max(data$HasRC.C) * (samples[["Depth"]] + samples[["r_noun[report,HasRC.C]"]]))
RTFactThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + max(data$HasRC.C) * (samples[["Depth"]] + samples[["r_noun[fact,HasRC.C]"]]))
FactReportDifferenceThree = RTFactThree - RTReportThree 
samples$FactReportDifferenceThree = FactReportDifferenceThree
cat("Fact/Report Difference in Three", mean(FactReportDifferenceThree), " ", quantile(FactReportDifferenceThree, 0.025), " ", quantile(FactReportDifferenceThree, 0.975), "\n")

# in One condition
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]
RTReportOne = exp(samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + min(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactOne = exp(samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + min(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
FactReportDifferenceOne = RTFactOne - RTReportOne
samples$FactReportDifferenceOne = FactReportDifferenceOne
cat("Fact/Report Difference within One", mean(FactReportDifferenceOne), " ", quantile(FactReportDifferenceOne, 0.025), " ", quantile(FactReportDifferenceOne, 0.975), "\n")



# in Embedded condition (Two/Three)
RTReportEmbedded = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactEmbedded = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
FactReportDifferenceEmbedded = RTFactEmbedded - RTReportEmbedded 
samples$FactReportDifferenceEmbedded = FactReportDifferenceEmbedded
cat("Fact/Report-Like Difference across Two/Three", mean(FactReportDifferenceEmbedded), " ", quantile(FactReportDifferenceEmbedded, 0.025), " ", quantile(FactReportDifferenceEmbedded, 0.975), "\n")

RTReportTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + min(data$HasRC.C) * (samples[["Depth"]] + 0))
RTFactTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + min(data$HasRC.C) * (samples[["Depth"]] + 0))
FactReportDifferenceTwo = RTFactTwo - RTReportTwo 
samples$FactReportDifferenceTwo = FactReportDifferenceTwo
cat("Fact/Report-Like Difference in Two", mean(FactReportDifferenceTwo), " ", quantile(FactReportDifferenceTwo, 0.025), " ", quantile(FactReportDifferenceTwo, 0.975), "\n")

RTReportThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + max(data$HasRC.C) * (samples[["Depth"]] + 0))
RTFactThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + max(data$HasRC.C) * (samples[["Depth"]] + 0))
FactReportDifferenceThree = RTFactThree - RTReportThree 
samples$FactReportDifferenceThree = FactReportDifferenceThree
cat("Fact/Report-Like Difference in Three", mean(FactReportDifferenceThree), " ", quantile(FactReportDifferenceThree, 0.025), " ", quantile(FactReportDifferenceThree, 0.975), "\n")



# in One condition
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]
RTReportOne = exp(samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + 0 + min(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactOne = exp(samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + 0 + min(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
FactReportDifferenceOne = RTFactOne - RTReportOne
samples$FactReportDifferenceOne = FactReportDifferenceOne
cat("Fact/Report-Like Difference within One", mean(FactReportDifferenceOne), " ", quantile(FactReportDifferenceOne, 0.025), " ", quantile(FactReportDifferenceOne, 0.975), "\n")


sink()
#samples$CompatibilityEffect_Two = CompatibilityEffect

plot = mcmc_areas(samples, pars=c("DepthEffect", "FactReportDifferenceOne", "FactReportDifferenceTwo", "FactReportDifferenceThree", "CompatibilityEffect_Two", "CompatibilityEffect_Three"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-RawEffects_Outliers.pdf", width=5, height=5)




#plot = mcmc_areas(samples, pars=c("DepthEffect", "FactReportDifference"), prob=.95, n_dens=32, adjust=5)
#ggsave(plot, file="figures/posterior-histograms-RawRTs_Outliers.pdf", width=5, height=4)


EmbeddingBias.C = c(EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact) #, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact)
#HasSC = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)-0.5
#HasRC = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)-0.5
HasSC.C = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)-0.8
HasRC.C = c(0.5, 0.5, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)-0.5
HasSC = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
HasRC = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)
compatible.C = c(0.5, 0.5, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE)-0.5
compatible = c(FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE)
#Compatible = c(0.5, 0.5, 0, 0, 1, 1, 0, 0, 1, 1)-0.5
RTPred = exp(mean(samples$b_Intercept) + HasSC.C * mean(samples$b_HasSC.C) + HasRC.C * mean(samples$b_HasRC.C) + EmbeddingBias.C * mean(samples$b_EmbeddingBias.C) + HasSC.C * EmbeddingBias.C * mean(samples[["Embedded:EmbeddingBias"]]) + HasRC.C * EmbeddingBias.C * mean(samples[["Depth:EmbeddingBias"]]) + compatible.C * mean(samples$b_compatible.C) + compatible.C*HasRC.C*mean(samples[["Depth:Compatible"]]) + compatible.C*EmbeddingBias.C*mean(samples[["b_compatible.C:EmbeddingBias.C"]]))

predictions = data.frame(EmbeddingBias.C=EmbeddingBias.C, HasSC.C=HasSC.C, HasRC.C=HasRC.C, RTPred=RTPred, HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias.C-mean(data$EmbeddingBias.C-data$EmbeddingBias), compatible=compatible, compatible.C=compatible.C)
library(ggplot2)
dataPlot = ggplot(predictions, aes(x=EmbeddingBias.C, y=RTPred, group=paste(HasSC.C, HasRC.C), color=paste(HasSC.C, HasRC.C))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC.C, HasRC.C, noun, EmbeddingBias.C) %>% summarise(rt=mean(rt)), aes(x=EmbeddingBias.C, y=rt))

predictions$condition = paste(predictions$HasSC, predictions$HasRC, predictions$compatible)
dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=log(RTPred), group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC))) + geom_smooth(se=F, method="lm", aes(linetype=compatible)) + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC), linetype=compatible), alpha=0.3) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-fit_Outliers.pdf", height=3.5, width=1.8)


#RTReportTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))
#RTFactTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))


# Now specifically take the fixed-effects parts
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]

RTFactTwo = mean(samples$b_Intercept) + 0.2 * mean(samples$b_HasSC.C) + HasRC.C * mean(samples$b_HasRC.C) + EmbeddingBiasC_fact * mean(samples$b_EmbeddingBias.C) + 0.2 * EmbeddingBiasC_fact * mean(samples[["Embedded:EmbeddingBias"]]) + (-0.5) * EmbeddingBiasC_fact * mean(samples[["Depth:EmbeddingBias"]])

#RTCompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + max(data$compatible.C) * samples[["Compatible"]] + min(data$HasRC.C) * samples[["Depth"]] + max(data$compatible.C) * min(data$HasRC.C) * samples[["Depth:Compatible"]])

# COMPATIBLE
RTCompReportTwo = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]] + max(data$compatible.C) * samples[["Compatible:EmbeddingBias"]]    ))  + min(data$HasRC.C) * (samples[["Depth"]] + max(data$compatible.C) * samples[["Depth:Compatible"]]) + max(data$compatible.C) * samples[["Compatible"]]
RTCompFactTwo = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) *  samples[["Depth:EmbeddingBias"]] + max(data$compatible.C) * samples[["Compatible:EmbeddingBias"]])) + min(data$HasRC.C) * (samples[["Depth"]] + max(data$compatible.C) * samples[["Depth:Compatible"]]) + max(data$compatible.C) * samples[["Compatible"]]

RTCompReportThree = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]] + max(data$compatible.C) * samples[["Compatible:EmbeddingBias"]])) + max(data$HasRC.C) * (samples[["Depth"]] + max(data$compatible.C) * samples[["Depth:Compatible"]]) + max(data$compatible.C) * samples[["Compatible"]]
RTCompFactThree = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]] + max(data$compatible.C) * samples[["Compatible:EmbeddingBias"]])) + max(data$HasRC.C) * (samples[["Depth"]] + max(data$compatible.C) * samples[["Depth:Compatible"]]) + max(data$compatible.C) * samples[["Compatible"]]


# INCOMPATIBLE
RTIncompReportTwo = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]] + min(data$compatible.C) * samples[["Compatible:EmbeddingBias"]]    ))  + min(data$HasRC.C) * (samples[["Depth"]] + min(data$compatible.C) * samples[["Depth:Compatible"]]) + min(data$compatible.C) * samples[["Compatible"]]
RTIncompFactTwo = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) *  samples[["Depth:EmbeddingBias"]] + min(data$compatible.C) * samples[["Compatible:EmbeddingBias"]])) + min(data$HasRC.C) * (samples[["Depth"]] + min(data$compatible.C) * samples[["Depth:Compatible"]]) + min(data$compatible.C) * samples[["Compatible"]]

RTIncompReportThree = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]] + min(data$compatible.C) * samples[["Compatible:EmbeddingBias"]])) + max(data$HasRC.C) * (samples[["Depth"]] + min(data$compatible.C) * samples[["Depth:Compatible"]]) + min(data$compatible.C) * samples[["Compatible"]]
RTIncompFactThree = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]] + min(data$compatible.C) * samples[["Compatible:EmbeddingBias"]])) + max(data$HasRC.C) * (samples[["Depth"]] + min(data$compatible.C) * samples[["Depth:Compatible"]]) + min(data$compatible.C) * samples[["Compatible"]]


# in One condition
RTReportOne = (samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + 0 + min(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactOne = (samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + 0 + min(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))



# REPORT
report_HasSC = c(FALSE, TRUE, TRUE, TRUE, TRUE)
report_HasRC = c(FALSE, FALSE, TRUE, FALSE, TRUE)
report_compatible = c(FALSE, FALSE, FALSE, TRUE, TRUE)
report_EmbeddingBias = EmbeddingBiasC_report-mean(data$EmbeddingBias.C-data$EmbeddingBias) + c(0,0,0,0,0)
report_RTPred = c(mean((RTReportOne)), mean(RTIncompReportTwo), mean(RTIncompReportThree), mean((RTCompReportTwo)), mean((RTCompReportThree)))
report_upper = report_RTPred+c(sd((RTReportOne)), sd(RTIncompReportTwo), sd(RTIncompReportThree), sd((RTCompReportTwo)), sd((RTCompReportThree)))
report_lower = report_RTPred-c(sd((RTReportOne)), sd(RTIncompReportTwo), sd(RTIncompReportThree), sd((RTCompReportTwo)), sd((RTCompReportThree)))


# FACT
fact_HasSC = c(FALSE, TRUE, TRUE, TRUE, TRUE)
fact_HasRC = c(FALSE, FALSE, TRUE, FALSE, TRUE)
fact_compatible = c(FALSE, FALSE, FALSE, TRUE, TRUE)
fact_EmbeddingBias = EmbeddingBiasC_fact-mean(data$EmbeddingBias.C-data$EmbeddingBias) + c(0,0,0,0,0)
fact_RTPred = c(mean((RTFactOne)), mean(RTIncompFactTwo), mean(RTIncompFactThree), mean((RTCompFactTwo)), mean((RTCompFactThree)))
fact_upper = fact_RTPred+c(sd((RTFactOne)), sd(RTIncompFactTwo), sd(RTIncompFactThree), sd((RTCompFactTwo)), sd((RTCompFactThree)))
fact_lower = fact_RTPred-c(sd((RTFactOne)), sd(RTIncompFactTwo), sd(RTIncompFactThree), sd((RTCompFactTwo)), sd((RTCompFactThree)))

HasSC = c(report_HasSC, fact_HasSC)
HasRC = c(report_HasRC, fact_HasRC)
compatible = c(report_compatible, fact_compatible)
EmbeddingBias = c(report_EmbeddingBias, fact_EmbeddingBias)
RTPred = c(report_RTPred, fact_RTPred)
upper = c(report_upper, fact_upper)
lower = c(report_lower, fact_lower)

predictionsPoints = data.frame(HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias, RTPred=RTPred, lower=lower, upper=upper)


dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=(RTPred), group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC))) + geom_smooth(se=F, method="lm", aes(linetype=compatible)) + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(exp(LogRT))), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC), linetype=compatible), alpha=0.3) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Reading Time (milliseconds)") + geom_errorbar(data=predictionsPoints, aes(x=EmbeddingBias, ymin=exp(lower), ymax=exp(upper)), width=0.3) + ylim(800, 1800)
ggsave(dataPlot, file="figures/logRT-points-fit_errorbars_noLogTransform_Outliers.pdf", height=3.5, width=1.8)





