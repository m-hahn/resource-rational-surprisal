
library(ggplot2)
library(dplyr)
library(tidyr)


Datapoints = 2733
data = read.csv("analyze_Model_IncludingNoTP_E12_R_AICs.tsv", sep="\t")


# This concerns nine settings with very high or very low delta, and lambda <1 (e.g., delta=0.95+lambda=0.5)
missingConfigurations = data %>% filter(N<Datapoints)
data = data %>% filter(N == Datapoints)
imputed = data.frame()
for(i in (1:nrow(missingConfigurations))) {
	delta = missingConfigurations$deletion_rate[[i]]
	lambda = missingConfigurations$predictability_weight[[i]]
	to_impute = data %>% filter(abs(deletion_rate-delta) <= 0.05, abs(predictability_weight-lambda)<=0.25) %>% summarise(AIC=mean(AIC), AICRaw=mean(AICRaw), AICFull=mean(AICFull), N=mean(N))
	to_impute$deletion_rate=delta
	to_impute$predictability_weight=lambda
	to_impute$Other=NA
	imputed = rbind(imputed, to_impute)
}

data = rbind(data, imputed)



meanAIC = 2674.119
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICRaw=mean(AICRaw)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AICRaw)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AICRaw")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_E12_Viz_R_AICRaw.pdf", height=2, width=5)



meanAIC = 2682.151
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_E12_Viz_R.pdf", height=2, width=5)


# This is when adding all the fixed effects. There is still an improvement due to adding resource-rational surprisal
meanAICFull = 2421.531
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICFull=mean(AICFull)-meanAICFull), aes(x=deletion_rate, y=predictability_weight, fill=AICFull)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AICFull")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_E12_Viz_R_AICFull.pdf", height=2, width=5)


meanAIC = 2674.119
plot = ggplot(data %>% filter(predictability_weight==1) %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICDifference=mean(AICRaw)-meanAIC) %>% group_by() %>% mutate(deletion_rate=20*(1-deletion_rate)), aes(x=deletion_rate, y=1, fill=AICDifference)) + geom_tile() + theme_classic() + scale_fill_gradient2() + xlab("Retention Rate") + ylab("") + guides(fill=guide_legend(title="AICDifference")) +  theme(axis.text.y=element_blank())
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_E12_Viz_R_AICRaw_Lambda1_Integer.pdf", height=1.5, width=5)

