library(tidyr)
library(dplyr)


library(ggplot2)

data = read.csv("conceptual.tsv", sep="\t")

data$compatible = grepl("_compatible", data$Condition)
data$HasSC = !grepl("NoSC", data$Condition)
data$HasRC = grepl("RC", data$Condition)

data$HasSCHasRC = (paste(data$HasSC, data$HasRC, sep="_"))


plot = ggplot(data %>% filter(Theory == "NoisyChannel") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Ratio) %>% summarise(Difficulty=mean(Difficulty)), aes(x=Ratio, y=Difficulty, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Difficulty") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + theme(axis.text = element_blank(), axis.ticks = element_blank()) + ylim(-0.5, 4)
ggsave(plot, file="figures/conceptual-predictions-noisy-channel.pdf", width=2, height=2)







plot = ggplot(data %>% filter(Theory == "Surprisal") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Ratio) %>% summarise(Difficulty=mean(Difficulty)), aes(x=Ratio, y=Difficulty, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Difficulty") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + theme(axis.text = element_blank(), axis.ticks = element_blank()) + ylim(-0.5, 1.5)
ggsave(plot, file="figures/conceptual-predictions-surprisal.pdf", width=2, height=2)



data[data$Ratio == 0,]$Ratio = log(0.7)

plot = ggplot(data %>% filter(Theory == "DLT") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Ratio) %>% summarise(Difficulty=mean(Difficulty)), aes(x=Ratio, y=Difficulty, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Difficulty") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) # + theme(axis.text = element_blank(), axis.ticks = element_blank())
plot = plot + ylim(-0.5, 5.5) 
ggsave(plot, file="figures/conceptual-predictions-dlt.pdf", width=1.8, height=1.8)







# Unclear whether this really is what those models would predict, as we don't have broad-coverage implementations covering this effect
plot = ggplot(data %>% filter(Theory == "CueRetrieval") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Ratio) %>% summarise(Difficulty=mean(Difficulty)), aes(x=Ratio, y=Difficulty, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Difficulty") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + theme(axis.text = element_blank(), axis.ticks = element_blank()) + ylim(0.5, 5.5)
ggsave(plot, file="figures/conceptual-predictions-cueRetrieval.pdf", width=2, height=2)


