library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data1 = read.csv("../../previous/study5_replication/Submiterator-master/all_trials.tsv", sep="\t") %>% mutate(distractor=NA, group=NULL)
data2 = read.csv("../../experiment1/Submiterator-master/trials_byWord.tsv", sep="\t") %>% mutate(workerid=workerid+1000)
data3 = read.csv("../../experiment2/Submiterator-master/trials-experiment2.tsv", sep="\t") %>% mutate(workerid=workerid+2000)

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


# Load corpus counts (COCA)
nounsCOCA = read.csv("../../../../materials/nouns/corpus_counts/COCA/results/results_counts4.py.tsv", sep="\t")                                                                                                            
nounsCOCA$Conditional_COCA = log(nounsCOCA$theNOUNthat/nounsCOCA$theNOUN)                                                                                                                                   
nounsCOCA$Marginal_COCA = log(nounsCOCA$theNOUN)
nounsCOCA$Joint_COCA = log(nounsCOCA$theNOUNthat)

data$Noun = data$noun

data = merge(data, nounsCOCA, by=c("Noun"))


data$Conditional_COCA.C = data$Conditional_COCA - mean(data$Conditional_COCA, na.rm=TRUE)
data$Marginal_COCA.C = data$Marginal_COCA - mean(data$Marginal_COCA, na.rm=TRUE)
data$Joint_COCA.C = data$Joint_COCA - mean(data$Joint_COCA, na.rm=TRUE)


# Load corpus counts (ukWac)
nounsukwac = read.csv("../../../../materials/nouns/corpus_counts/ukwac/results/results_counts4.py.tsv", sep="\t")                                                                                                          
nounsukwac$Conditional_ukwac = log(nounsukwac$theNOUNthat/nounsukwac$theNOUN)                                                                                                                               
nounsukwac$Marginal_ukwac = log(nounsukwac$theNOUN)
nounsukwac$Joint_ukwac = log(nounsukwac$theNOUNthat)

data = merge(data, nounsukwac, by=c("Noun"))
 
data$Conditional_ukwac.C = data$Conditional_ukwac - mean(data$Conditional_ukwac, na.rm=TRUE)
data$Marginal_ukwac.C = data$Marginal_ukwac - mean(data$Marginal_ukwac, na.rm=TRUE)
data$Joint_ukwac.C = data$Joint_ukwac - mean(data$Joint_ukwac, na.rm=TRUE)

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
data = data %>% mutate(EmbeddingBias = Conditional_ukwac)
data = data %>% mutate(EmbeddingBias.C = Conditional_ukwac.C)

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
model = (brm(LogRT ~ trial + HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C + HasSC.C * EmbeddingBias.C + compatible.C * EmbeddingBias.C + (1+compatible.C+HasSC.C+HasRC.C+HasRC.C*compatible.C|noun) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C + HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C|workerid) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C +HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C|item), data=data %>% filter(Region == "REGION_3_0"), cores=4))

sink("output/analyze.R_ukwac.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze.R_fixed_ukwac.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$Depth = samples$b_HasRC.C
samples$EmbeddingBias = samples$b_EmbeddingBias.C
samples$Embedded = samples$b_HasSC.C
samples$Compatible = samples$b_compatible.C
plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Compatible"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects_ukwac.pdf", width=5, height=5)

samples[["Depth:EmbeddingBias"]] = samples[["b_HasRC.C:EmbeddingBias.C"]]
samples[["Embedded:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:HasSC.C"]]
samples[["Depth:Compatible"]] = samples[["b_HasRC.C:compatible.C"]]
samples[["Compatible:EmbeddingBias"]] = samples[["b_compatible.C:EmbeddingBias.C"]]
plot = mcmc_areas(samples, pars=c("Depth:EmbeddingBias", "Embedded:EmbeddingBias", "Depth:Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-interactions_ukwac.pdf", width=5, height=5)


plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Compatible", "Depth:EmbeddingBias", "Embedded:EmbeddingBias", "Depth:Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms_ukwac.pdf", width=5, height=4)


embeddingBiasSamples = data.frame(EmbeddingBiasWithinOne = samples$b_EmbeddingBias.C + min(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasAcrossTwoThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasWithinTwo = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + -5.630822e-01 * samples[["b_HasRC.C:EmbeddingBias.C"]], EmbeddingBiasWithinThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + 4.369178e-01 * samples[["b_HasRC.C:EmbeddingBias.C"]])
plot = mcmc_areas(embeddingBiasSamples, prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-EmbeddingBias_ukwac.pdf", width=5, height=5)


sink("output/analyze.R_posteriors_ukwac.txt")
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

RTReportTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))
RTFactTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))
FactReportDifferenceTwo = RTFactTwo - RTReportTwo 
samples$FactReportDifferenceTwo = FactReportDifferenceTwo
cat("Fact/Report Difference in Two", mean(FactReportDifferenceTwo), " ", quantile(FactReportDifferenceTwo, 0.025), " ", quantile(FactReportDifferenceTwo, 0.975), "\n")

RTReportThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))
RTFactThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))
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


sink()
#samples$CompatibilityEffect_Two = CompatibilityEffect

plot = mcmc_areas(samples, pars=c("FactReportDifferenceOne", "FactReportDifferenceTwo", "FactReportDifferenceThree"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-RawEffects_EmbeddingBias_ukwac.pdf", width=5, height=3)



