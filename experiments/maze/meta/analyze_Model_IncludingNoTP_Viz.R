
library(ggplot2)
library(dplyr)
library(tidyr)



data = read.csv("analyze_Model_IncludingNoTP_R_AICs.tsv", sep="\t") %>% filter(N == 5332)

meanAIC = 5298.719
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICRaw=mean(AICRaw)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AICRaw)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AICRaw")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_Viz_R_AICRaw.pdf", height=2, width=5)



meanAIC = 5300.835
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_Viz_R.pdf", height=2, width=5)


# This is when adding all the fixed effects. There is still an improvement due to adding resource-rational surprisal
meanAICFull = 4678.892
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICFull=mean(AICFull)-meanAICFull), aes(x=deletion_rate, y=predictability_weight, fill=AICFull)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AICFull")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_Viz_R_AICFull.pdf", height=2, width=5)


meanAIC = 5298.719
plot = ggplot(data %>% filter(predictability_weight==1) %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICDifference=mean(AICRaw)-meanAIC) %>% group_by() %>% mutate(deletion_rate=20*(1-deletion_rate)), aes(x=deletion_rate, y=1, fill=AICDifference)) + geom_tile() + theme_classic() + scale_fill_gradient2() + xlab("Retention Rate") + ylab("") + guides(fill=guide_legend(title="AICDifference")) +  theme(axis.text.y=element_blank())
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_Viz_R_AICRaw_Lambda1_Integer.pdf", height=1.5, width=5)



