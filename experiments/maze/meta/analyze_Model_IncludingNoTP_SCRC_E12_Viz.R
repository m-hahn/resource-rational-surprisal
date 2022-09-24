
library(ggplot2)
library(dplyr)
library(tidyr)


Datapoints = 2095
data = read.csv("analyze_Model_IncludingNoTP_SCRC_R_AICs.tsv", sep="\t")


# This concerns nine settings with very high or very low delta, and lambda <1 (e.g., delta=0.95+lambda=0.5)
data = data %>% filter(N == Datapoints)
doneConfigurations = paste(data$deletion_rate, data$predictability_weight)
deltas = c()
lambdas = c()
for(delta in unique(data$deletion_rate)) {
	for(lambda in unique(data$predictability_weight)) {
		if(!(paste(delta, lambda) %in% doneConfigurations)) {
			deltas = c(deltas, delta)
			lambdas = c(lambdas, lambda)
		}
	}
}
missingConfigurations = data.frame(deletion_rate = deltas, predictability_weight = lambdas)

imputed = data.frame()
for(i in (1:nrow(missingConfigurations))) {
	delta = missingConfigurations$deletion_rate[[i]]
	lambda = missingConfigurations$predictability_weight[[i]]
	to_impute = data %>% filter(abs(deletion_rate-delta) <= 0.05, abs(predictability_weight-lambda)<=0.25) %>% summarise(AIC=mean(AIC), AICRaw=mean(AICRaw), AICFull=mean(AICFull), N=mean(N))
	to_impute$deletion_rate=delta
	to_impute$predictability_weight=lambda
	to_impute$None=NA
	imputed = rbind(imputed, to_impute)
}

data = rbind(data, imputed)




meanAICRaw = 2387.221
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICRaw=mean(AICRaw)-meanAICRaw), aes(x=deletion_rate, y=predictability_weight, fill=AICRaw)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_SCRC_Viz_R_Raw.pdf", height=2, width=5)

meanAIC = 2394.282
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_SCRC_Viz_R.pdf", height=2, width=5)

meanAICFull = 2365.662
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AICFull=mean(AICFull)-meanAICFull), aes(x=deletion_rate, y=predictability_weight, fill=AICFull)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("Predictability Weight") + guides(fill=guide_legend(title="AICFull")) + scale_y_reverse ()
ggsave(plot, file="figures/analyze_Model_IncludingNoTP_SCRC_Viz_R_Full.pdf", height=2, width=5)



