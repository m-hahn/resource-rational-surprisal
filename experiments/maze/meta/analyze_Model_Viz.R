
library(ggplot2)
library(dplyr)
library(tidyr)



data = read.csv("analyze_Model_R_AICs.tsv", sep="\t") %>% filter(N == 5332)

meanAIC = 5298.741
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=1, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_Viz_R.pdf", height=2, width=5)


# This is when adding all the fixed effects. There is still an improvement due to adding resource-rational surprisal
meanAICFull = 4678.733
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICFull=mean(AICFull)-meanAICFull), aes(x=deletion_rate, y=1, fill=AICFull)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("") + guides(fill=guide_legend(title="AICFull")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_Viz_R_AICFull.pdf", height=2, width=5)


