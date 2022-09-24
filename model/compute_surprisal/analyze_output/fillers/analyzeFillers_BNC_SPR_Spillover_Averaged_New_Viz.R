
library(ggplot2)
library(dplyr)
library(tidyr)


Datapoints = 26401
data = read.csv("output/analyzeFillers_freq_BNC_SPR_Spillover_Averaged_New_R.tsv", sep="\t")

# This concerns nine settings with very high or very low delta, and lambda <1 (e.g., delta=0.95+lambda=0.5)
data = data %>% filter(NData == Datapoints)
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
	to_impute = data %>% filter(abs(deletion_rate-delta) <= 0.05, abs(predictability_weight-lambda)<=0.25) %>% summarise(AIC=mean(AIC), NData=mean(NData))
	to_impute$deletion_rate=delta
	to_impute$predictability_weight=lambda
	to_impute$Coefficient=NA
	to_impute$Correlation=NA
	to_impute$X=NA
	imputed = rbind(imputed, to_impute)
}

data = rbind(data, imputed)



# Viz for all parameter configurations
meanAIC = 16938.34
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse ()
ggsave(plot, file="figures/analyzeFillers_freq_BNC_SPR_Spillover_Averaged_New_R.pdf", height=2, width=5)

# Viz for lambda=1
meanAIC = 16938.34
plot = ggplot(data %>% filter(predictability_weight==1) %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC), aes(x=deletion_rate, y=predictability_weight, fill=AIC)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("") + guides(fill=guide_legend(title="AIC")) + scale_y_reverse () + theme(legend.position="bottom")
ggsave(plot, file="figures/analyzeFillers_freq_BNC_SPR_Spillover_Averaged_New_R_Lambda1.pdf", height=1.5, width=5)


# Viz for lambda=1
meanAIC = 16938.34
plot = ggplot(data %>% filter(predictability_weight==1) %>% group_by(deletion_rate, predictability_weight) %>% summarise(AIC=mean(AIC)-meanAIC) %>% mutate(AICDifference=pmin(AIC, 50)), aes(x=20*(1-deletion_rate), y=predictability_weight, fill=AICDifference)) + geom_tile() + theme_classic() + scale_fill_gradient2() + xlab("Average Retention Rate") + ylab("") + guides(fill=guide_legend(title="AIC")) +  theme(legend.position="bottom") + theme(axis.text.y=element_blank())
ggsave(plot, file="figures/analyzeFillers_freq_BNC_SPR_Spillover_Averaged_New_R_Lambda1_Integer.pdf", height=1.5, width=5)

