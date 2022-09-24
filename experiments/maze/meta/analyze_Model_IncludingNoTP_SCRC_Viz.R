
library(ggplot2)
library(dplyr)
library(tidyr)



data = read.csv("analyze_Model_IncludingNoTP_SCRC_R_AICs.tsv", sep="\t") %>% filter(N == 2095)

meanAICRaw = 2387.221
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICRaw=mean(AICRaw)-meanAICRaw), aes(x=deletion_rate, y=predictability_weight, fill=AICRaw)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_SCRC_Viz_R_Raw.pdf", height=2, width=5)

meanAIC = 2394.282
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_SCRC_Viz_R.pdf", height=2, width=5)

meanAICFull = 2365.662
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICFull=mean(AICFull)-meanAICFull), aes(x=deletion_rate, y=predictability_weight, fill=AICFull)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AICFull")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_SCRC_Viz_R_Full.pdf", height=2, width=5)



