library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lme4)
library(dplyr)

# Read all model predictions
model = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3_Fillers.py.tsv", sep="\t")

# Column indicating the position of each word in the setence
model = model %>% mutate(wordInItem = as.numeric(as.character(Region))+1)

# Create item IDs corresponding to those in the human data
model = model %>% mutate(item = paste("Filler", Sentence, sep="_"))

# Average over repeat runs of the importance sampler. `SurprisalReweighted' is the estimate of Resource-Rational Lossy-Context Surprisal computed by the importance sampler.
model = model %>% group_by(item, wordInItem, Region, Word, Script, ID, predictability_weight, deletion_rate)
model = model %>% summarise(SurprisalReweighted=mean(as.numeric(as.character(SurprisalReweighted)), na.rm=TRUE))
# variables appearing here:
#  item: the ID of the filler sentence
#  wordInItem, Region: position of word in sentence
#  Word: the word
#  Script: script used for training the model. Here, only those containing "_TPS" are used, the others reflect earlier versions of the model (see next line).
#  ID: ID of the model run
#  predictability_weight: only take those with value 1
#  deletion_rate: what fraction of the last N=20 words is forgotton on average. Only 0.05 is of interest here (for now, look at almost perfect memory)
model = model %>% filter(grepl("_TPS", Script), predictability_weight==1)
model = model %>% filter(deletion_rate==0.05, predictability_weight==1)
# Now, average across all model runs
model = model %>% group_by(wordInItem, item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE))

# Some further processing
model$LowerCaseToken = tolower(model$Word)
model$WordLength = nchar(as.character(model$Word))


# COCA word frequencies
word_freq_50000 = read.csv("stimuli-coca-frequencies.tsv", sep="\t", quote=NULL)
word_freq_50000$LogWordFreq_COCA = log(word_freq_50000$Frequency)
model = merge(model, word_freq_50000, by=c("LowerCaseToken"), all=TRUE)

# BNC word frequencies
word_freq_50000 = read.csv("stimuli-bnc-frequencies.tsv", sep="\t", quote=NULL)
word_freq_50000$LogWordFreq = log(word_freq_50000$Frequency)
model = merge(model, word_freq_50000, by=c("LowerCaseToken"), all=TRUE)

# Residualize COCA on BNC
model$LogWordFreq_COCA.R = resid(lm(LogWordFreq_COCA~LogWordFreq, data=model, na.action=na.exclude))

# Load human data from Experiments 1 and 2
data1 = read.csv("../../../../../experiments/maze/experiment1/Submiterator-master/trials_byWord.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid)
data2 = read.csv("../../../../../experiments/maze/experiment2/Submiterator-master/trials-experiment2.tsv",sep="\t",quote="@") %>% mutate(workerid = workerid+1000)
rts = rbind(data1, data2)
# We could also add the data from  ../../../../../experiments/maze/previous/study5_replication/Submiterator-master/all_trials.tsv


# Removing extreme values. Might actually prefer not to do this here.
rts = rts[rts$rt < quantile(rts$rt, 0.99),]
rts = rts[rts$rt > quantile(rts$rt, 0.01),]

# Remove an extraneous "232_" that arises in some items/experiments
rts = rts %>% mutate(item = str_replace(item, "232_", ""))
# Remove critical and practice items
rts = rts %>% select(wordInItem, item, workerid, rt, word) %>% filter(!grepl("Mixed", item), !grepl("Practice", item), !grepl("Critical", item), wordInItem != 0) # the model has no surprisal prediction for the first word



model$itemID = paste(model$item, model$wordInItem, sep="_")


# Merge human data and model predictions
      data = merge(model, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq))

library(ggplot2)

# Create visualizations
plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise(rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=SurprisalReweighted, y=rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot1.pdf", height=10, width=10)

plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise(log_rt = mean(log(rt)), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=SurprisalReweighted, y=log_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot2.pdf", height=10, width=10)


plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise(rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=exp(-SurprisalReweighted), y=rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot3.pdf", height=10, width=10)

plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( sd_rt = sd(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=SurprisalReweighted, y=sd_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot4.pdf", height=10, width=10)


plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( sd_rt = sd(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=rt, y=sd_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot4b.pdf", height=10, width=10)

plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( sd_log_rt = sd(log(rt)), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=rt, y=sd_log_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot4c.pdf", height=10, width=10)


library(moments)
plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( kurtosis_rt = kurtosis(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=SurprisalReweighted, y=kurtosis_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot5.pdf", height=10, width=10)

plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( kurtosis_rt = kurtosis(rt), sd_rt = sd(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=sd_rt, y=kurtosis_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot5b.pdf", height=10, width=10)


plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( kurtosis_rt = kurtosis(rt), sd_rt = sd(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=rt, y=kurtosis_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot5c.pdf", height=10, width=10)



#plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( kurtosis_rt = kurtosis(rt), sd_rt = sd(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=rt, y=kurtosis_rt)) + geom_smooth() + geom_point() + geom_line(data=(data %>% group_by(wordInItem, word) %>% summarise(MeanRT = mean(rt), SHAPE=(mean(rt)/sd(rt))^2, SCALE=sd(rt)^2/mean(rt))) %>% mutate(gamma_kurtosis=3+6/SHAPE), aes(x=MeanRT, y=gamma_kurtosis), color="red") + geom_line(data=(data %>% group_by(wordInItem, word) %>% summarise(MeanRT = mean(rt), MU=mean(log(rt)), SIGMA=sd(log(rt)))) %>% mutate(lognormal_kurtosis=(exp(4*sigma^2) + 2*exp(3*sigma^2)+3*exp(2*sigma^2)-3)), aes(x=MeanRT, y=lognormal_kurtosis), color="green")




plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( skewness_rt = skewness(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=SurprisalReweighted, y=skewness_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot6.pdf", height=10, width=10)


plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( skewness_rt = skewness(rt), sd_rt = sd(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=sd_rt, y=skewness_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot6b.pdf", height=10, width=10)


plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise( skewness_rt = skewness(rt), sd_rt = sd(rt), rt = mean(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=rt, y=skewness_rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot6c.pdf", height=10, width=10)


plot = ggplot(data %>% group_by(wordInItem, item) %>% summarise(rt = median(rt), SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE), LogWordFreq=mean(LogWordFreq, na.rm=TRUE)), aes(x=SurprisalReweighted, y=rt)) + geom_smooth() + geom_point()
ggsave(plot, file="figures/surprisal-rts-plot7.pdf", height=10, width=10)



plot = ggplot(data %>% filter(item == "Filler_0"), aes(x=rt)) + geom_histogram() + facet_wrap(~wordInItem)
ggsave(plot, file="figures/surprisal-rts-plot8.pdf", height=10, width=10)


library(lme4)
model = lmer(rt ~ (1|workerid), data=data)

data$rtResid = 972.39 + resid(model)

plot = ggplot(data %>% filter(item == "Filler_0"), aes(x=rtResid)) + geom_histogram() + facet_wrap(~wordInItem)
ggsave(plot, file="figures/surprisal-rts-plot9.pdf", height=10, width=10)

plot = ggplot(data %>% filter(item == "Filler_0"), aes(x=rt)) + geom_density() + facet_wrap(~wordInItem)
ggsave(plot, file="figures/surprisal-rts-plot10.pdf", height=10, width=10)


# Here, I was trying to fit Gamma and Lognormal distributions. Another candidate is Ex-Gaussian.

data_0 = data %>% filter(item == "Filler_0")
plot = ggplot(data_0, aes(x=rt)) + geom_histogram() + stat_function(data=data_0 %>% group_by(wordInItem) %>% summarise(SHAPE=(mean(rt)/sd(rt))^2, SCALE=sd(rt)^2/mean(rt)), fun=dgamma, args=list(shape=SHAPE, rate=1/SCALE)) + facet_wrap(~wordInItem)
ggsave(plot, file="figures/surprisal-rts-plot11.pdf", height=10, width=10)

data_0 = data %>% filter(item == "Filler_0")
plot = ggplot(data_0, aes(x=rt)) + geom_density() + geom_line(data=merge(data_0 %>% group_by(wordInItem, word) %>% summarise(SHAPE=(mean(rt)/sd(rt))^2, SCALE=sd(rt)^2/mean(rt)), data.frame(X=(1:4000))) %>% mutate(gamma_density=dgamma(X, shape=SHAPE, scale=SCALE)), aes(x=X, y=gamma_density), color="red") + geom_line(data=merge(data_0 %>% group_by(wordInItem, word) %>% summarise(MU=mean(log(rt)), SIGMA=sd(log(rt))), data.frame(X=(1:4000))) %>% mutate(lognormal_density=dlnorm(X, MU, SIGMA)), aes(x=X, y=lognormal_density), color="green") + facet_wrap(~wordInItem+word)
ggsave(plot, file="figures/surprisal-rts-plot12.pdf", height=10, width=10)

data_0 = data %>% filter(item == "Filler_1")
plot = ggplot(data_0, aes(x=rt)) + geom_density() + geom_line(data=merge(data_0 %>% group_by(wordInItem, word) %>% summarise(SHAPE=(mean(rt)/sd(rt))^2, SCALE=sd(rt)^2/mean(rt)), data.frame(X=(1:4000))) %>% mutate(gamma_density=dgamma(X, shape=SHAPE, scale=SCALE)), aes(x=X, y=gamma_density), color="red") + geom_line(data=merge(data_0 %>% group_by(wordInItem, word) %>% summarise(MU=mean(log(rt)), SIGMA=sd(log(rt))), data.frame(X=(1:4000))) %>% mutate(lognormal_density=dlnorm(X, MU, SIGMA)), aes(x=X, y=lognormal_density), color="green") + facet_wrap(~wordInItem+word)
ggsave(plot, file="figures/surprisal-rts-plot13.pdf", height=10, width=10)

data_0 = data %>% filter(item == "Filler_2")
plot = ggplot(data_0, aes(x=rt)) + geom_density() + geom_line(data=merge(data_0 %>% group_by(wordInItem, word) %>% summarise(SHAPE=(mean(rt)/sd(rt))^2, SCALE=sd(rt)^2/mean(rt)), data.frame(X=(1:4000))) %>% mutate(gamma_density=dgamma(X, shape=SHAPE, scale=SCALE)), aes(x=X, y=gamma_density), color="red") + geom_line(data=merge(data_0 %>% group_by(wordInItem, word) %>% summarise(MU=mean(log(rt)), SIGMA=sd(log(rt))), data.frame(X=(1:4000))) %>% mutate(lognormal_density=dlnorm(X, MU, SIGMA)), aes(x=X, y=lognormal_density), color="green") + facet_wrap(~wordInItem+word)
ggsave(plot, file="figures/surprisal-rts-plot14.pdf", height=10, width=10)


#
## TODO look at n't --> breaks tokenization
#sink("analyzeFillers_freq_BNC_Spillover_R.tsv")
#cat("predictability_weight", "deletion_rate", "ID", "NData", "AIC", "Coefficient", "\n", sep="\t")
#sink()
#
#for(ID_ in unique(model$ID)) {
#   if(TRUE) { #!(ID_  %in% alreadyDone)) {
#      data = model %>% filter(ID == ID_)
#      data = merge(data, rts, by=c("wordInItem", "item")) %>% filter(!is.na(LogWordFreq))
#      if(nrow(data) > 0) {
#         data$SurprisalReweighted = resid(lm(SurprisalReweighted ~ LogWordFreq, data=data))
#         data$LogRT = log(data$rt)
#         dataPrevious = data %>% mutate(wordInItem=wordInItem+1)
#         data = merge(data, dataPrevious, by=c("wordInItem", "item", "workerid", "predictability_weight", "deletion_rate")) # "sentence",
#         lmermodel = lmer(LogRT.x ~ SurprisalReweighted.x + wordInItem + LogWordFreq.x + LogWordFreq_COCA.R.x + WordLength.x + SurprisalReweighted.y + LogWordFreq.y + LogWordFreq_COCA.R.y + WordLength.y + LogRT.y + (1|itemID.x) + (1|workerid), data=data)
#         cat(mean(data$predictability_weight), mean(data$deletion_rate), ID_, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
#         sink("analyzeFillers_freq_BNC_Spillover_R.tsv", append=TRUE)
#         cat(mean(data$predictability_weight), mean(data$deletion_rate), ID_, nrow(data), AIC(lmermodel), coef(summary(lmermodel))[2,1], "\n", sep="\t")
#         sink()
#      }
#   }
#}
#
