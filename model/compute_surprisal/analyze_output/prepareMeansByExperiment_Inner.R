library(dplyr)
library(tidyr)
data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3_Inner.py.tsv", sep="\t") %>% filter(!grepl("245_", Item), Region == "V2_0")

print(unique(data$Item))
print("Items")
#write.table(data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted)), file="prepareMeansByExperiment.R.tsv", sep="\t")
byItemData = data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=round(mean(Surprisal), 3), SurprisalReweighted=round(mean(SurprisalReweighted), 3), ThatFraction=round(mean(ThatFraction), 3), ThatFractionReweighted=round(mean(ThatFractionReweighted), 3))
byItemData = byItemData[order(byItemData$ID, byItemData$Noun, byItemData$ID, byItemData$Region),]
write.table(byItemData, file="prepareMeansByExperiment_ByStimuli_Inner.R.tsv", sep="\t", quote=FALSE)


