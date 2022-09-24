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

# Code the nouns and frames
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



# Load corpus counts (Wikipedia)
nounFreqs = read.csv("~/forgetting/corpus_counts/wikipedia/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)

nounFreqs2 = read.csv("~/forgetting/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)

# Code Embedding Bias
data = data %>% mutate(EmbeddingBias = True_False_False-False_False_False)
data = data %>% mutate(EmbeddingBias.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))


# Log-Transform Reading Times
data$LogRT = log(data$rt)

# Center trial order
data$trial = data$trial - mean(data$trial, na.rm=TRUE)


# Mixed-Effects Analysis
library(brms)
model = (brm(LogRT ~ EmbeddingBias.C + (1|noun) + (1 + EmbeddingBias.C |workerid) + (1 + EmbeddingBias.C|item), data=data %>% filter(RegionFine == "VP1_0")))

library(ggrepel)
dataPlot = ggplot(data %>% filter(RegionFine == "VP1_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(exp(LogRT))), aes(x=EmbeddingBias, y=rt)) + geom_smooth(method="lm") + geom_text_repel(aes(label=noun)) + theme_bw() + xlab("Embedding Bias") + ylab("Reading Time")
ggsave(dataPlot, file="figures/rt-raw.pdf", height=3.5, width=3.5)



sink("output/analyze.R.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analyze.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$EmbeddingBias = samples$b_EmbeddingBias.C
plot = mcmc_areas(samples, pars=c("EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-main_effects.pdf", width=5, height=2)

ggsave(plot, file="figures/posterior-histograms.pdf", width=5, height=2)


embeddingBiasSamples = data.frame(EmbeddingBiasWithinThree = samples$b_EmbeddingBias.C)
plot = mcmc_areas(embeddingBiasSamples, prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-EmbeddingBias.pdf", width=5, height=2)

EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]

sink("output/analyze.R_posteriors.txt")
RTReportThree = exp(samples$b_Intercept + samples[["r_noun[report,Intercept]"]] + EmbeddingBiasC_report * (samples[["EmbeddingBias"]] ))
RTFactThree = exp(samples$b_Intercept + samples[["r_noun[fact,Intercept]"]] + EmbeddingBiasC_fact * (samples[["EmbeddingBias"]] ))
FactReportDifferenceThree = RTFactThree - RTReportThree 
samples$FactReportDifferenceThree = FactReportDifferenceThree
cat("Fact/Report Difference in Three", mean(FactReportDifferenceThree), " ", quantile(FactReportDifferenceThree, 0.025), " ", quantile(FactReportDifferenceThree, 0.975), "\n")
sink()

plot = mcmc_areas(samples, pars=c("FactReportDifferenceThree"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms-RawRTs.pdf", width=5, height=2)


EmbeddingBias.C = c(EmbeddingBiasC_report, EmbeddingBiasC_fact)
HasSC.C = c(TRUE, TRUE)
HasRC.C = c(TRUE, TRUE)-0.5
HasSC = c(TRUE, TRUE)
HasRC = c(TRUE, TRUE)
compatible.C = c(FALSE, FALSE)-0.5
compatible = c(FALSE, FALSE)
RTPred = exp(mean(samples$b_Intercept) + EmbeddingBias.C * mean(samples$b_EmbeddingBias.C))

predictions = data.frame(EmbeddingBias.C=EmbeddingBias.C, HasSC.C=HasSC.C, HasRC.C=HasRC.C, RTPred=RTPred, HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias.C-mean(data$EmbeddingBias.C-data$EmbeddingBias), compatible=compatible, compatible.C=compatible.C)
library(ggplot2)
dataPlot = ggplot(predictions, aes(x=EmbeddingBias.C, y=RTPred, group=paste(HasSC.C, HasRC.C), color=paste(HasSC.C, HasRC.C))) + geom_smooth(method="lm") + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC.C, HasRC.C, noun, EmbeddingBias.C) %>% summarise(rt=mean(rt)), aes(x=EmbeddingBias.C, y=rt))

predictions$condition = paste(predictions$HasSC, predictions$HasRC, predictions$compatible)
dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=log(RTPred), group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC))) + geom_smooth(se=F, method="lm", aes(linetype=compatible)) + geom_point(data=data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC), linetype=compatible), alpha=0.3) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Log Reading Time")
ggsave(dataPlot, file="figures/logRT-points-fit.pdf", height=3.5, width=1.8)


samples$b_HasSC.C = 0
samples$b_HasRC.C = 0
samples[["Embedded:EmbeddingBias"]] = 0
samples [["Depth:EmbeddingBias"]] = 0

# Now specifically take the fixed-effects parts
EmbeddingBiasC_report = (data %>% filter(noun == "report"))$EmbeddingBias.C[[1]]
EmbeddingBiasC_fact = (data %>% filter(noun == "fact"))$EmbeddingBias.C[[1]]

RTFactTwo = mean(samples$b_Intercept) + 0.2 * mean(samples$b_HasSC.C) + HasRC.C * mean(samples$b_HasRC.C) + EmbeddingBiasC_fact * mean(samples$b_EmbeddingBias.C) + 0.2 * EmbeddingBiasC_fact * mean(samples[["Embedded:EmbeddingBias"]]) + (-0.5) * EmbeddingBiasC_fact * mean(samples[["Depth:EmbeddingBias"]])

#RTCompatible = exp(samples$b_Intercept + max(data$HasSC.C) * samples[["Embedded"]] + max(data$compatible.C) * samples[["Compatible"]] + min(data$HasRC.C) * samples[["Depth"]] + max(data$compatible.C) * min(data$HasRC.C) * samples[["Depth:Compatible"]])


samples[["Embedded"]] = 0
	samples[["Compatible:EmbeddingBias"]] = 0
	samples[["Depth:Compatible"]] = 0
	samples[["Compatible"]] = 0

data$compatible.C = 1
data$HasRC.C = 1
data$HasSC.C = 1
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
report_HasSC = c(TRUE)
report_HasRC = c(TRUE)
report_compatible = c(FALSE)
report_EmbeddingBias = EmbeddingBiasC_report-mean(data$EmbeddingBias.C-data$EmbeddingBias) + c(0,0,0,0,0)
report_RTPred = c(mean(RTIncompReportThree))
report_upper = report_RTPred+c(sd(RTIncompReportThree))
report_lower = report_RTPred-c(sd(RTIncompReportThree))


# FACT
fact_HasSC = c(TRUE)
fact_HasRC = c(TRUE)
fact_compatible = c(FALSE)
fact_EmbeddingBias = EmbeddingBiasC_fact-mean(data$EmbeddingBias.C-data$EmbeddingBias) + c(0,0,0,0,0)
fact_RTPred = c(mean(RTIncompReportThree))
fact_upper = fact_RTPred+c(sd(RTIncompReportThree))
fact_lower = fact_RTPred-c(sd(RTIncompReportThree))


HasSC = c(report_HasSC, fact_HasSC)
HasRC = c(report_HasRC, fact_HasRC)
compatible = c(report_compatible, fact_compatible)
EmbeddingBias = c(report_EmbeddingBias, fact_EmbeddingBias)
RTPred = c(report_RTPred, fact_RTPred)
upper = c(report_upper, fact_upper)
lower = c(report_lower, fact_lower)

predictionsPoints = data.frame(HasSC=HasSC, HasRC=HasRC, EmbeddingBias=EmbeddingBias, RTPred=RTPred, lower=lower, upper=upper)

data$HasSC = TRUE
data$HasRC = TRUE
data$compatible=FALSE

dataPlot = ggplot(predictions, aes(x=(EmbeddingBias), y=(RTPred), group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC))) + geom_smooth(se=F, method="lm", aes(linetype=compatible)) + geom_point(data=data %>% filter(RegionFine == "VP1_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias, compatible) %>% summarise(rt=mean(exp(LogRT))), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC, compatible), color=paste(HasSC, HasRC), linetype=compatible), alpha=0.3) +  scale_color_manual(values = c("FALSE FALSE" = "#F8766D", "TRUE FALSE"="#00BA38", "TRUE TRUE"="#619CFF")) + theme_bw() + theme(legend.position="none") + xlab("Embedding Bias") + ylab("Reading Time") + geom_errorbar(data=predictionsPoints, aes(x=EmbeddingBias, ymin=exp(lower), ymax=exp(upper)), width=0.3) + ylim(800, 1700)
ggsave(dataPlot, file="figures/logRT-points-fit_errorbars_noLogTransform.pdf", height=3.5, width=1.8)


#dataPlot = ggplot(data %>% filter(Region == "REGION_3_0") %>% group_by(HasSC, HasRC, noun, EmbeddingBias) %>% summarise(rt=mean(LogRT)), aes(x=EmbeddingBias, y=rt, group=paste(HasSC, HasRC), color=paste(HasSC, HasRC))) +  geom_point()

#+ scale_color_manual(values = c("FALSE_FALSE" = "#F8766D", "TRUE_FALSE"="#00BA38", "TRUE_TRUE"="#619CFF"))


