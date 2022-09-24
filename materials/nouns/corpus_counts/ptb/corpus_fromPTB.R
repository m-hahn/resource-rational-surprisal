data = read.csv("corpus/results/results_counts4.py.tsv", sep="\t")


library(dplyr)
library(tidyr)

data$LCount = log(1+data$Count)

data$Condition = paste(data$HasThat, data$Capital, sep="_")

byNoun = as.data.frame(unique(data) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount))

write.table(byNoun[order(byNoun$True_False),], file="fromCorpus_counts.csv", sep="\t", quote=FALSE, row.names=FALSE)


data2 = read.csv("verbForgettingLogOdds_Joint.csv") %>% rename(Noun = noun) %>% mutate(X=NULL)

byNoun = merge(byNoun, data2, by=c("Noun"), all.x=TRUE)



byNoun = byNoun %>% select(Noun, True_False, ForgettingVerbLogOdds)
write.table(byNoun[order(byNoun$True_False),], file="fromCorpus.csv", sep="\t", quote=FALSE, row.names=FALSE)

