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
model = (brm(LogRT ~ EmbeddingBias.C + (1|noun) + (1 + EmbeddingBias.C |workerid) + (1 + EmbeddingBias.C|item), data=data %>% filter(RegionFine == "REGION_V1_0")))

library(ggrepel)
dataPlot = ggplot(data %>% filter(RegionFine == "REGION_V1_0") %>% group_by(noun, EmbeddingBias) %>% summarise(rt=mean(exp(LogRT))), aes(x=EmbeddingBias, y=rt)) + geom_smooth(method="lm") + geom_text_repel(aes(label=noun)) + theme_bw() + xlab("Embedding Bias") + ylab("Reading Time") + ylim(800, 1550)
ggsave(dataPlot, file="figures/rt-raw.pdf", height=5.5, width=5.5)



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

#plot = mcmc_areas(samples, pars=c("DepthEffect", "FactReportDifference"), prob=.95, n_dens=32, adjust=5)
#ggsave(plot, file="figures/posterior-histograms-RawRTs.pdf", width=5, height=4)

