library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data = read.csv("trials_byWord.tsv",sep="\t")
# Note:
# There are 99 subjects in the recorded data. Evidently, data for 1 subject failed to be uploaded to the server.



# Exclude participants with excessive error rates
participantsByErrorsBySlide = data %>% filter(correct != "none") %>% group_by(workerid) %>% summarise(ErrorsBySlide = mean(correct == "no"))
data = merge(data, participantsByErrorsBySlide, by=c("workerid"))
data = data %>% filter(ErrorsBySlide < 0.2)


# Remove trials with incorrect responses
data = data %>% filter(rt > 0, correct == "yes")

# Remove extremely low or extremely high reading times
data = data %>% filter(rt < quantile(data$rt, 0.99))
data = data %>% filter(rt > quantile(data$rt, 0.01))


# Only consider critical trials
data = data %>% filter(condition != "filler")


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
data = data %>% mutate(HasSC = ifelse(condition == "critical_NoSC", FALSE, TRUE))
data = data %>% mutate(HasRC = ifelse(condition %in% c("critical_SCRC_compatible", "critical_SCRC_incompatible"), TRUE, FALSE))

# Contrast Coding
data$HasRC.C = ifelse(data$HasRC, 0.5, ifelse(data$HasSC, -0.5, 0)) #resid(lm(HasRC ~ HasSC, data=data))
data$HasSC.C = data$HasSC - 0.8

# Log-Transform Reading Times
data$LogRT = log(data$rt)

# Center trial order
data$trial = data$trial - mean(data$trial, na.rm=TRUE)


# Mixed-Effects Analysis
library(brms)
model = (brm(LogRT ~ HasRC.C + HasRC.C * EmbeddingBias.C + HasSC.C * EmbeddingBias.C + EmbeddingBias.C + (1+HasSC.C+HasRC.C|noun) + (1 + EmbeddingBias.C + HasSC.C + HasSC.C * EmbeddingBias.C + HasRC.C+ HasRC.C * EmbeddingBias.C|workerid) + (1+EmbeddingBias.C + HasSC.C +HasSC.C * EmbeddingBias.C + HasRC.C +  HasRC.C + HasRC.C * EmbeddingBias.C|item), data=data %>% filter(Region == "REGION_3_0"), cores=4, iterations=8000))

sink("output/analyze.R.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$Depth = samples$b_HasRC.C
samples$EmbeddingBias = samples$b_EmbeddingBias.C
samples$Embedded = samples$b_HasSC.C
plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects.pdf", width=5, height=5)

samples[["Depth:EmbeddingBias"]] = samples[["b_HasRC.C:EmbeddingBias.C"]]
samples[["Embedded:EmbeddingBias"]] = samples[["b_EmbeddingBias.C:HasSC.C"]]
plot = mcmc_areas(samples, pars=c("Depth:EmbeddingBias", "Embedded:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-interactions.pdf", width=5, height=5)


plot = mcmc_areas(samples, pars=c("Depth", "EmbeddingBias", "Embedded", "Depth:EmbeddingBias", "Embedded:EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms.pdf", width=5, height=4)


embeddingBiasSamples = data.frame(EmbeddingBiasWithinOne = samples$b_EmbeddingBias.C + min(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasAcrossTwoThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]], EmbeddingBiasWithinTwo = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + (-0.5) * samples[["b_HasRC.C:EmbeddingBias.C"]], EmbeddingBiasWithinThree = samples$b_EmbeddingBias.C + max(data$HasSC.C) * samples[["b_EmbeddingBias.C:HasSC.C"]] + 0.5 * samples[["b_HasRC.C:EmbeddingBias.C"]])
plot = mcmc_areas(embeddingBiasSamples, prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-EmbeddingBias.pdf", width=5, height=5)


sink("output/analyze.R_posteriors.txt")
cat("b_HasRC.C ", mean(samples$b_HasRC.C<0), "\n")
cat("b_EmbeddingBias.C:HasSC.C ", mean(samples[["b_EmbeddingBias.C:HasSC.C"]]>0), "\n")
cat("EmbeddingBiasAcrossTwoThree ", mean(embeddingBiasSamples$EmbeddingBiasAcrossTwoThree>0), "\n")
RTThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + max(data$HasRC.C) * samples[["Depth"]])
RTTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + min(data$HasRC.C) * samples[["Depth"]])
DepthEffect = RTThree-RTTwo
samples$DepthEffect = DepthEffect
cat("Effect Depth", mean(DepthEffect), " ", quantile(DepthEffect, 0.025), " ", quantile(DepthEffect, 0.975), "\n")


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

plot = mcmc_areas(samples, pars=c("DepthEffect", "FactReportDifferenceOne", "FactReportDifferenceTwo", "FactReportDifferenceThree"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-RawEffects.pdf", width=5, height=3)


#plot = mcmc_areas(samples, pars=c("DepthEffect", "FactReportDifference"), prob=.95, n_dens=32, adjust=5)
#ggsave(plot, file="figures/posterior-histograms-RawRTs.pdf", width=5, height=4)


EmbeddingBias.C = c(EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact) #, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_report, EmbeddingBiasC_fact)
#HasSC = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)-0.5
#HasRC = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)-0.5
HasSC.C = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)-0.8
HasRC.C = c(0.5, 0.5, FALSE, FALSE, TRUE, TRUE)-0.5
HasSC = c(FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)
HasRC = c(FALSE, FALSE, FALSE, FALSE, TRUE, TRUE)
#Compatible = c(0.5, 0.5, 0, 0, 1, 1, 0, 0, 1, 1)-0.5
RTPred = exp(mean(samples$b_Intercept) + HasSC.C * mean(samples$b_HasSC.C) + HasRC.C * mean(samples$b_HasRC.C) + EmbeddingBias.C * mean(samples$b_EmbeddingBias.C) + HasSC.C * EmbeddingBias.C * mean(samples[["Embedded:EmbeddingBias"]]) + HasRC.C * EmbeddingBias.C * mean(samples[["Depth:EmbeddingBias"]]))

#RTReportThree = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]) + max(data$HasRC.C) * samples[["Depth"]] + max(data$HasRC.C) * (samples[["Depth"]] + 0))       


predictions = data.frame(EmbeddingBias.C=EmbeddingBias.C, HasSC.C=HasSC.C, HasRC.C=HasRC.C, RTPred=RTPred, HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias.C-mean(data$EmbeddingBias.C-data$EmbeddingBias))
library(ggplot2)
dataPlot = ggplot(predictions, aes(x=EmbeddingBias.C, y=RTPred, group=paste(HasSC.C, HasRC.C), color=paste(HasSC.C, HasRC.C))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC.C, HasRC.C, noun, EmbeddingBias.C) %>% summarise(rt=mean(rt)), aes(x=EmbeddingBias.C, y=rt))

dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=log(RTPred), group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-fit.pdf", height=3.5, width=1.8)


#RTReportTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[report,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[report,HasSC.C]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))
#RTFactTwo = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + samples[["r_noun[fact,Intercept]"]] + max(data$HasSC.C) * samples[["r_noun[fact,HasSC.C]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))


# Now specifically take the fixed-effects parts
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]

RTFactTwo = mean(samples$b_Intercept) + 0.2 * mean(samples$b_HasSC.C) + HasRC.C * mean(samples$b_HasRC.C) + EmbeddingBiasC_fact * mean(samples$b_EmbeddingBias.C) + 0.2 * EmbeddingBiasC_fact * mean(samples[["Embedded:EmbeddingBias"]]) + (-0.5) * EmbeddingBiasC_fact * mean(samples[["Depth:EmbeddingBias"]])


RTReportTwo = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) * samples[["Depth:EmbeddingBias"]]))  + min(data$HasRC.C) * samples[["Depth"]]
RTFactTwo = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + min(data$HasRC.C) *  samples[["Depth:EmbeddingBias"]])) + min(data$HasRC.C) * samples[["Depth"]]

RTReportThree = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]])) + max(data$HasRC.C) * samples[["Depth"]]
RTFactThree = (samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + 0 + max(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + max(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]] + max(data$HasRC.C) * samples[["Depth:EmbeddingBias"]])) + max(data$HasRC.C) * samples[["Depth"]]

# in One condition
RTReportOne = (samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + 0 + min(data$HasSC.C) * 0 + EmbeddingBiasC_report * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))
RTFactOne = (samples$b_Intercept + min(data$HasSC.C) * samples[["Embedded"]] + 0 + min(data$HasSC.C) * 0 + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]]  + min(data$HasSC.C) * samples[["Embedded:EmbeddingBias"]]))




HasSC = c(FALSE, TRUE, TRUE, FALSE, TRUE, TRUE)
HasRC = c(FALSE, FALSE, TRUE, FALSE, FALSE, TRUE)
EmbeddingBias = c(EmbeddingBiasC_report, EmbeddingBiasC_report, EmbeddingBiasC_report, EmbeddingBiasC_fact, EmbeddingBiasC_fact, EmbeddingBiasC_fact)-mean(data$EmbeddingBias.C-data$EmbeddingBias)
RTPred = c(mean((RTReportOne)), mean((RTReportTwo)), mean((RTReportThree)), mean((RTFactOne)), mean((RTFactTwo)), mean((RTFactThree)))
upper = RTPred+c(sd((RTReportOne)), sd((RTReportTwo)), sd((RTReportThree)), sd((RTFactOne)), sd((RTFactTwo)), sd((RTFactThree)))
lower = RTPred-c(sd((RTReportOne)), sd((RTReportTwo)), sd((RTReportThree)), sd((RTFactOne)), sd((RTFactTwo)), sd((RTFactThree)))
predictionsPoints = data.frame(HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias, RTPred=RTPred, lower=lower, upper=upper)
dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=log(RTPred), group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC), color=paste(HasSC, HasRC)), alpha=0.5) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time") + geom_errorbar(data=predictionsPoints, aes(x=EmbeddingBias, ymin=lower, ymax=upper), width=0.3) + ylim(6.7, 7.55)
ggsave(dataPlot, file="figures/logRT-points-fit_errorbars.pdf", height=3.5, width=1.8)

dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=(RTPred), group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias) %>% summarise(rt=mean(exp(LogRT))), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC), color=paste(HasSC, HasRC)), alpha=0.5) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Reading Time (milliseconds)") + geom_errorbar(data=predictionsPoints, aes(x=EmbeddingBias, ymin=exp(lower), ymax=exp(upper)), width=0.3) + ylim(800, 1600)
ggsave(dataPlot, file="figures/logRT-points-fit_errorbars_noLogTransform.pdf", height=3.5, width=1.8)




