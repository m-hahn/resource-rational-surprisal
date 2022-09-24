library(dplyr)
library(tidyr)
data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t") %>% filter(!grepl("245_", Item), Region == "V1_0")
# Note: there is an error message about embedded nul(s). This is caused by some abnormality that happened in writing model prediction in model ID 16540137, potentially two processes writing at the same time or premature termination. This can be avoided by adding skipNul = TRUE as an argument to read.csv, which removes a single (malformed) line in the resulting TSV file.

print(unique(data$Item))
print("Items")
#write.table(data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted)), file="prepareMeansByExperiment.R.tsv", sep="\t")
byItemData = data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted))
byItemData = byItemData[order(byItemData$ID, byItemData$Noun, byItemData$ID, byItemData$Region),]
write.table(byItemData, file="prepareMeansByExperiment_ByStimuli.R.tsv", sep="\t")


