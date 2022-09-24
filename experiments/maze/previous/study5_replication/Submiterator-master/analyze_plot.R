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
nounFreqs = read.csv("~/forgetting/corpus_counts/wikipedia/results/results_counts4NEW.py.tsv", sep="\t")
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


plot = ggplot(data = data %>% filter(Region == "REGION_3_0") %>% group_by(noun, condition, EmbeddingBias) %>% summarise(rt = mean(rt)), aes(x=EmbeddingBias, y=rt, group=condition, color=condition)) + geom_smooth(method="lm", se=F) + geom_point() + theme_bw() + theme(legend.position="none")
plot = ggplot(data = data %>% filter(Region == "REGION_3_0"), aes(x=EmbeddingBias, y=rt, group=condition, color=condition)) + geom_smooth(method="lm", se=F) + theme_bw() + theme(legend.position="none")
ggsave(plot, file="figures/rt-CriticalVerb_StudyS5.pdf", width=2, height=4)

plot = ggplot(data = data %>% filter(Region == "REGION_3_0") %>% group_by(noun, condition, EmbeddingBias) %>% summarise(LogRT = mean(LogRT)), aes(x=EmbeddingBias, y=LogRT, group=condition, color=condition)) + geom_smooth(method="lm", se=F) + geom_point() + theme_bw() + theme(legend.position="none")

plot = ggplot(data = data %>% filter(Region == "REGION_3_0"), aes(x=EmbeddingBias, y=LogRT, group=condition, color=condition)) + geom_smooth(method="lm", se=F) + theme_bw() + theme(legend.position="none")
ggsave(plot, file="figures/logrt-CriticalVerb_StudyS5.pdf", width=2, height=4)



