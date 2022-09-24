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

plot = ggplot(participantsByErrorsBySlide, aes(x=ErrorsBySlide)) + geom_histogram() + theme_bw() + xlab("Fraction of Trials (Slides) with Error")
ggsave(plot, file="figures/slides-errors_prior.pdf", width=3, height=3)

plot = ggplot(data %>% mutate(condition_group = ifelse(condition == "filler", "Filler", "Critical")) %>% group_by(condition_group, wordInItem) %>% summarise(error_rate=mean(correct == "no")), aes(x=wordInItem, y=error_rate, group=condition_group, color=condition_group)) + geom_line()
plot = ggplot(data %>% group_by(wordInItem) %>% summarise(error_rate=mean(correct == "no")), aes(x=wordInItem, y=error_rate)) + geom_line() + xlab("Word Number") + ylab("Error Rate")
ggsave(plot, file="figures/errors-by-position_prior.pdf", width=5, height=3)

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
model = (brm(LogRT ~ HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C + HasSC.C * EmbeddingBias.C + compatible.C * EmbeddingBias.C + (1+compatible.C+HasSC.C+HasRC.C+HasRC.C*compatible.C|noun) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C + HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C|workerid) + (1+compatible.C + EmbeddingBias.C + compatible.C * EmbeddingBias.C + HasSC.C +HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C * compatible.C + HasRC.C * EmbeddingBias.C|item), data=data %>% filter(Region == "REGION_3_0"), cores=4, iter=2000, prior=c(prior("normal(0,1)", class="b"), prior("student_t(3,0,2.5)", class="sigma"), prior("student_t(3,0,2.5)", class="sd"), prior("student_t(3,7,2.5)", class="Intercept"))))


sink("output/analyze.R_prior.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze.R_fixed_prior.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$Depth = samples$b_HasRC.C
samples$EmbeddingBias = samples$b_EmbeddingBias.C
samples$Embedded = samples$b_HasSC.C
samples$Compatible = samples$b_compatible.C
plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Compatible"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects_prior.pdf", width=5, height=5)

samples[["Depth:EmbeddingBias"]] = samples[["b_HasRC.C:EmbeddingBias.C"]]
samples[["Embedded:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:HasSC.C"]]
samples[["Depth:Compatible"]] = samples[["b_HasRC.C:compatible.C"]]
samples[["Compatible:EmbeddingBias"]] = samples[["b_compatible.C:EmbeddingBias.C"]]
plot = mcmc_areas(samples, pars=c("Depth:EmbeddingBias", "Embedded:EmbeddingBias", "Depth:Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-interactions_prior.pdf", width=5, height=5)


plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Compatible", "Depth:EmbeddingBias", "Embedded:EmbeddingBias", "Depth:Compatible", "Compatible:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms_prior.pdf", width=5, height=4)


embeddingBiasSamples = data.frame(EmbeddingBiasWithinOne = samples$b_EmbeddingBias.C + min(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasAcrossTwoThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasWithinTwo = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + -5.630822e-01 * samples[["b_HasRC.C:EmbeddingBias.C"]], EmbeddingBiasWithinThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + 4.369178e-01 * samples[["b_HasRC.C:EmbeddingBias.C"]])
plot = mcmc_areas(embeddingBiasSamples, prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-EmbeddingBias_prior.pdf", width=5, height=5)


sink("output/analyze.R_posteriors_prior.txt")
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
ggsave(plot, file="figures/posterior-histograms-RawEffects_prior.pdf", width=5, height=5)







items = unique(data$item)


slopes = data.frame(item=c(), embBias=c(), compatibility=c())

for(item_ in items) {
#item_ = "Critical_4"

slope_embBias = samples[["b_EmbeddingBias.C"]] + samples[[paste("r_item[", item_, ",EmbeddingBias.C]", sep="")]] + max(data$HasSC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",EmbeddingBias.C:HasSC.C]", sep="")]] + samples[["b_EmbeddingBias.C:HasSC.C"]])



slope_embBias_one = samples[["b_EmbeddingBias.C"]] + samples[[paste("r_item[", item_, ",EmbeddingBias.C]", sep="")]] + min(data$HasSC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",EmbeddingBias.C:HasSC.C]", sep="")]] + samples[["b_EmbeddingBias.C:HasSC.C"]]) 

slope_embBias_two = samples[["b_EmbeddingBias.C"]] + samples[[paste("r_item[", item_, ",EmbeddingBias.C]", sep="")]] + max(data$HasSC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",EmbeddingBias.C:HasSC.C]", sep="")]] + samples[["b_EmbeddingBias.C:HasSC.C"]]) + min(data$HasRC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",EmbeddingBias.C:HasRC.C]", sep="")]] + samples[["Depth:EmbeddingBias"]])


slope_embBias_three = samples[["b_EmbeddingBias.C"]] + samples[[paste("r_item[", item_, ",EmbeddingBias.C]", sep="")]] + max(data$HasSC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",EmbeddingBias.C:HasSC.C]", sep="")]] + samples[["b_EmbeddingBias.C:HasSC.C"]]) + max(data$HasRC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",EmbeddingBias.C:HasRC.C]", sep="")]] + samples[["b_HasRC.C:EmbeddingBias.C"]])


slope_comp = samples[["b_compatible.C"]] + samples[[paste("r_item[", item_, ",compatible.C]", sep="")]]
slope_comp_two = samples[["b_compatible.C"]] + samples[[paste("r_item[", item_, ",compatible.C]", sep="")]] + min(data$HasRC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",compatible.C:HasRC.C]", sep="")]] + samples[["b_HasRC.C:compatible.C"]])
slope_comp_three = samples[["b_compatible.C"]] + samples[[paste("r_item[", item_, ",compatible.C]", sep="")]] + max(data$HasRC.C, na.rm=TRUE) * (samples[[paste("r_item[", item_, ",compatible.C:HasRC.C]", sep="")]] + samples[["b_HasRC.C:compatible.C"]])
slope_depth = samples[["b_HasRC.C"]] + samples[[paste("r_item[", item_, ",HasRC.C]", sep="")]] 
intercept = samples[["b_Intercept"]] + samples[[paste("r_item[", item_, ",Intercept]", sep="")]] 
slopes = rbind(slopes, data.frame(item=c(item_), embBias=c(mean(slope_embBias)), compatibility=c(mean(slope_comp)), embBias_one = c(mean(slope_embBias_one)), embBias_two = c(mean(slope_embBias_two)), embBias_three = c(mean(slope_embBias_three)), compatibility_two = c(mean(slope_comp_two)), compatibility_three = c(mean(slope_comp_three)), depth=c(mean(slope_depth)), intercept=c(mean(intercept))))

}

write.table(slopes, file="output/analyze.R_slopes_prior.tsv", sep="\t")


 item_ = "232_Critical_10"



print(get_prior(model))

