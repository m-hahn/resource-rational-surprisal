library(dplyr)
library(tidyr)
data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E1.py.tsv", sep="\t") %>% filter(!(Item %in% c("Mixed_0", "Mixed_10", "Mixed_20", "ItemMixed_22", "ItemMixed_27", "ItemMixed_28", "ItemMixed_30", "ItemMixed_32")))

write.table(data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted)), file="prepareMeansByExperiment_E1.R.tsv", sep="\t")
byItemData = data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted))
byItemData = byItemData[order(byItemData$ID, byItemData$Noun, byItemData$ID, byItemData$Region),]
write.table(byItemData, file="prepareMeansByExperiment_E1_ByStimuli.R.tsv", sep="\t")


