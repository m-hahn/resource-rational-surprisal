
library(ggplot2)
library(dplyr)
library(tidyr)


data = read.csv("output/extractSlopes_Fives_WithOne_All.py.tsv", sep="\t")


stimuli0 = read.csv("../../../../../materials/stimuli/tsv/RTExperimentsPrevious.tsv", sep="\t") %>% mutate(Sentence=NULL) %>% filter(Experiment == "Removed")
stimuli1 = read.csv("../../../../../materials/stimuli/tsv/Experiment1.tsv", sep="\t")
stimuli2 = read.csv("../../../../../materials/stimuli/tsv/Experiment2.tsv", sep="\t")

stimuli = rbind(stimuli0, stimuli1, stimuli2)

library(stringr)
stimuli = merge(stimuli %>% rename(Item=ID), data %>% mutate(Item=str_replace(item, "Item", "")), by=c("Item"))

stimuli = stimuli %>% filter(Item != "Item238_Critical_VAdv1")


meanN = function(x) {
	return(mean(x, na.rm=TRUE))
}

stimuli = stimuli %>% filter(Experiment == "E2") %>% filter(lambda==1, delta>=0.2, delta<0.8) %>% group_by(Item) %>% summarise(intercept=meanN(intercept), embedding=meanN(embedding), embBias_One=meanN(embBias_One), compatible_Two=meanN(compatible_Two), compatible_Three=meanN(compatible_Three), compatible.EmbBias=meanN(compatible.EmbBias), depth=meanN(depth), embBias_Two=meanN(embBias_Two), embBias_Three=meanN(embBias_Three))

stimuli = stimuli %>% filter(!is.na(embedding))

EmbeddingBias = c(-5, -0.3)

REPORT =  (-4.8--2.258807)
FACT =  (-0.3--2.258807)


# The abolute values, obtained using this method, are a bit lower than when averaging by-noun means. Not clear whether this is due to item/noun variation or a bug in the slope extraction script. I suspect it's the latter, because the values are consistently lower than the little pale dots in the per-noun figures. But the pattern is the same.

pred_One_Report = (stimuli$intercept) - 0.5 * (stimuli$embedding) + REPORT * (stimuli$embBias_One)
mean(pred_One_Report)
sd(pred_One_Report)/sqrt(length(pred_One_Report))

pred_One_Fact = (stimuli$intercept) - 0.5 * (stimuli$embedding) + FACT * (stimuli$embBias_One)
mean(pred_One_Fact)
sd(pred_One_Fact)/sqrt(length(pred_One_Fact))

pred_Two_Report_Compatible = stimuli$intercept + 0.5 * stimuli$embedding - 0.5 * stimuli$depth + REPORT * stimuli$embBias_Two + 0.5 * stimuli$compatible_Two + (0.5*REPORT) * stimuli$compatible.EmbBias
mean(pred_Two_Report_Compatible)
sd(pred_Two_Report_Compatible)/sqrt(length(pred_Two_Report_Compatible))


pred_Two_Fact_Compatible = stimuli$intercept + 0.5 * stimuli$embedding - 0.5 * stimuli$depth + FACT * stimuli$embBias_Two + 0.5 * stimuli$compatible_Two + (0.5*FACT) * stimuli$compatible.EmbBias
mean(pred_Two_Fact_Compatible)
sd(pred_Two_Fact_Compatible)/sqrt(length(pred_Two_Fact_Compatible))


pred_Three_Report_Compatible = stimuli$intercept + 0.5 * stimuli$embedding + 0.5 * stimuli$depth + REPORT * stimuli$embBias_Three + 0.5 * stimuli$compatible_Three + (0.5*REPORT) * stimuli$compatible.EmbBias
mean(pred_Three_Report_Compatible)
sd(pred_Three_Report_Compatible)/sqrt(length(pred_Three_Report_Compatible))


pred_Three_Fact_Compatible = stimuli$intercept + 0.5 * stimuli$embedding + 0.5 * stimuli$depth + FACT * stimuli$embBias_Three + 0.5 * stimuli$compatible_Three + (0.5*FACT) * stimuli$compatible.EmbBias
mean(pred_Three_Fact_Compatible)
sd(pred_Three_Fact_Compatible)/sqrt(length(pred_Three_Fact_Compatible))


pred_Two_Report_Incompatible = stimuli$intercept + 0.5 * stimuli$embedding - 0.5 * stimuli$depth + REPORT * stimuli$embBias_Two + -0.5 * stimuli$compatible_Two + (-0.5*REPORT) * stimuli$compatible.EmbBias
mean(pred_Two_Report_Incompatible)
sd(pred_Two_Report_Incompatible)/sqrt(length(pred_Two_Report_Incompatible))


pred_Two_Fact_Incompatible = stimuli$intercept + 0.5 * stimuli$embedding - 0.5 * stimuli$depth + FACT * stimuli$embBias_Two + -0.5 * stimuli$compatible_Two + (-0.5*FACT) * stimuli$compatible.EmbBias
mean(pred_Two_Fact_Incompatible)
sd(pred_Two_Fact_Incompatible)/sqrt(length(pred_Two_Fact_Incompatible))


pred_Three_Report_Incompatible = stimuli$intercept + 0.5 * stimuli$embedding + 0.5 * stimuli$depth + REPORT * stimuli$embBias_Three + -0.5 * stimuli$compatible_Three + (-0.5*REPORT) * stimuli$compatible.EmbBias
mean(pred_Three_Report_Incompatible)
sd(pred_Three_Report_Incompatible)/sqrt(length(pred_Three_Report_Incompatible))


pred_Three_Fact_Incompatible = stimuli$intercept + 0.5 * stimuli$embedding + 0.5 * stimuli$depth + FACT * stimuli$embBias_Three + -0.5 * stimuli$compatible_Three + (-0.5*FACT) * stimuli$compatible.EmbBias
mean(pred_Three_Fact_Incompatible)
sd(pred_Three_Fact_Incompatible)/sqrt(length(pred_Three_Fact_Incompatible))


