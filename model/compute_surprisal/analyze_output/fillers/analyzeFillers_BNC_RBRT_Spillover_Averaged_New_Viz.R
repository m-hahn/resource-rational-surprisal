:
library(ggplot2)
library(dplyr)
library(tidyr)



data = read.csv("output/analyzeFillers_freq_BNC_RBRT_Spillover_Averaged_New_R.tsv", sep="\t") %>% filter(NData == 7271)

meanAIC = 94877.55
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyzeFillers_freq_BNC_RBRT_Spillover_Averaged_New_R.pdf", height=2, width=5)


