

library(dplyr)
library(tidyr)
library(ggplot2)
data = read.csv("assembleHyperparameterResults.py.tsv", sep="\t", col.names=c("lambda", "delta", "RealRetentionRate", "Reward", "Surprisal", "ReconstructionLoss", "lr_memory", "lr_autoencoder", "lr_lm", "momentum", "ID", "Script")) %>% filter(grepl("S.py", Script))
data[data$Surprisal==5,]$Surprisal=NA
data$lambda = as.factor(data$lambda)


plot = ggplot(data, aes(x=delta, y=RealRetentionRate)) + geom_point() + geom_line(data=data.frame(x=c(0,1), y=c(1,0)), aes(x=x, y=y))


plot = ggplot(data %>% filter(!grepl("TPS.py", Script)) %>% group_by(delta, lambda) %>% summarise(Surprisal=mean(Surprisal)), aes(x=delta, y=Surprisal, group=lambda, color=lambda)) + geom_line() + theme_classic()
ggsave(plot, file="figures/delta-lambda-surprisal.pdf", height=3, width=5)

plot = ggplot(data %>% filter(!grepl("TPS.py", Script)) %>% group_by(delta, lambda) %>% summarise(ReconstructionLoss=mean(ReconstructionLoss)), aes(x=delta, y=ReconstructionLoss, group=lambda, color=lambda)) + geom_line() + theme_classic()
ggsave(plot, file="figures/delta-lambda-reconstruction.pdf", height=3, width=5)


plot = ggplot(data %>% filter(!grepl("TPS.py", Script)), aes(x=Surprisal, y=ReconstructionLoss, group=lambda, color=lambda)) + geom_smooth() + geom_point() + theme_classic()









