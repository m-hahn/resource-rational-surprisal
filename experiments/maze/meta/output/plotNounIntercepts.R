data = read.csv("extractPerNounIntercepts_Raw.R.tsv", sep="\t")

library(ggplot2)
library(tidyr)
library(dplyr)

data = data[order(data$embBias),]
data$noun = factor(data$noun, levels=data$noun)

#plot = ggplot(data, aes(y=noun, x=estimate)) + geom_errorbarh(aes(xmin=lower, xmax=upper))

library(ggrepel)
#plot = ggplot(data, aes(y=embBias, x=estimate)) + geom_label_repel(aes(label=noun)) + geom_errorbarh(aes(xmin=lower, xmax=upper)) + theme_bw()


#plot = ggplot(data, aes(y=embBias, x=estimate)) + geom_smooth(method="loess") + geom_point() + geom_errorbarh(aes(xmin=lower, xmax=upper)) + theme_bw()


#plot = ggplot(data, aes(x=embBias-2.3, y=estimate)) + geom_smooth(method="lm") + geom_point() + theme_bw()

library(ggpubr)
plot = ggplot(data, aes(x=embBias-2.3, y=estimate)) + geom_smooth(method="lm") + geom_text_repel(aes(label=noun)) + theme_bw() + stat_cor(label.x=-3, label.y=0.15, size=7) + xlab("Embedding Bias") + ylab("Reading Time Effect in Two/Three")
ggsave(plot, file="plotNounIntercepts_R.pdf", height=6, width=6)



library(tidyr)
library(dplyr)
model_E2 = read.csv("../../../../model/compute_surprisal/analyze_output/prepareMeansByExperiment_ByStimuli.R.tsv", quote='"', sep="\t") %>% mutate(Experiment = "Experiment2")
model_E1 = read.csv("../../../../model/compute_surprisal/analyze_output/prepareMeansByExperiment_E1_ByStimuli.R.tsv", quote='"', sep="\t") %>% mutate(Experiment = "Experiment1")
model = rbind(model_E1, model_E2)
library(ggplot2)
library(lme4)
model$compatible = grepl("_co", model$Condition)
model$HasSC = !grepl("NoSC", model$Condition)
model$HasRC = grepl("RC", model$Condition)
model$HasSCHasRC = (paste(model$HasSC, model$HasRC, sep="_"))

model_ = model %>% filter(predictability_weight == 1, deletion_rate==0.5, grepl("_TPL", Script), Region == "V1_0", HasRC, compatible) %>% group_by(compatible, HasRC, Noun) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))

model__ = lmer(SurprisalReweighted ~ compatible + (1|Noun), data=model_)

plot = ggplot(u, aes(x=SurprisalReweighted, y=estimate)) + geom_smooth(method="lm") + geom_text_repel(aes(label=noun)) + theme_bw() + stat_cor(label.x=8, label.y=0.15, size=7) + xlab("Embedding Bias") + ylab("Reading Time Effect in Two/Three")
ggsave(plot, file="plotNounIntercepts_R_SurprisalRT.pdf", height=6, width=6)



#
## the full raw predictions
#plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
#                                "TRUE_FALSE"="#00BA38",
#                                "TRUE_TRUE"="#619CFF")) 
#
#


#cor.test(data$embBias, data$estimate)
#cor.test(data$embBias, data$estimate, method="spearman")
#
#
#
#
#
#data2 = data[ data$embBias-2.3>-3,]
#nrow(data2)
#cor.test(data2$embBias, data2$estimate)
#
#
#plot = ggplot(data, aes(x=embBias, y=estimate)) + geom_smooth(method="lm") + geom_label_repel(aes(label=noun)) + theme_bw()
#
#data$Noun = data$noun
#
#
#nounsCOCA = read.csv("../../../../../../forgetting/corpus_counts/COCA/results/results_counts4.py.tsv", sep="\t")                                                                                                            
#nounsCOCA$Conditional_COCA = log(nounsCOCA$theNOUNthat/nounsCOCA$theNOUN)                                                                                                                                   
#nounsCOCA$Marginal_COCA = log(nounsCOCA$theNOUN)
#nounsCOCA$Joint_COCA = log(nounsCOCA$theNOUNthat)
#
#data = merge(data, nounsCOCA, by=c("Noun"))
#
#
#data$Conditional_COCA.C = data$Conditional_COCA - mean(data$Conditional_COCA, na.rm=TRUE)
#data$Marginal_COCA.C = data$Marginal_COCA - mean(data$Marginal_COCA, na.rm=TRUE)
#data$Joint_COCA.C = data$Joint_COCA - mean(data$Joint_COCA, na.rm=TRUE)
#
#nounsukwac = read.csv("../../../../../../forgetting/corpus_counts/ukwac/results/results_counts4.py.tsv", sep="\t")                                                                                                          
#nounsukwac$Conditional_ukwac = log(nounsukwac$theNOUNthat/nounsukwac$theNOUN)                                                                                                                               
#nounsukwac$Marginal_ukwac = log(nounsukwac$theNOUN)
#nounsukwac$Joint_ukwac = log(nounsukwac$theNOUNthat)
#
#data = merge(data, nounsukwac, by=c("Noun"))
#
#
#plot = ggplot(data, aes(x=embBias, y=estimate)) + geom_smooth(method="lm") + geom_label_repel(aes(label=noun)) + theme_bw()
#plot = ggplot(data, aes(x=Conditional_ukwac, y=estimate)) + geom_smooth(method="lm") + geom_label_repel(aes(label=noun)) + theme_bw()
#plot = ggplot(data, aes(x=Conditional_COCA, y=estimate)) + geom_smooth(method="lm") + geom_label_repel(aes(label=noun)) + theme_bw()
#
#
#plot = ggplot(data, aes(y=Conditional_ukwac, x=estimate)) + geom_label_repel(aes(label=noun)) + geom_errorbarh(aes(xmin=lower, xmax=upper)) + theme_bw()
#
#plot = ggplot(data, aes(y=Conditional_ukwac, x=estimate)) + geom_smooth(method="loess") + geom_point() + geom_errorbarh(aes(xmin=lower, xmax=upper)) + theme_bw()
#
