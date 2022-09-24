library(tidyr)
library(dplyr)
library(lme4)


# Read trial data
data = read.csv("trials.tsv", sep="\t")

# Read annotation of completions
annotated = read.csv("annotated_embedded.tsv", sep="\t")
data = merge(data, annotated, by=c("completion"), all=FALSE)

# Select critical trials
data = data %>% filter(condition == "SC_RC")

library(stringr)

noun = c()
remainder = c()
sentences = as.character(data$sentence)
matrices = c()
for(i in (1:nrow(data))) {
        strin = strsplit(str_replace(sentences[i], "  ", " "), " ")
        noun = c(noun, strsplit(strin[[1]][6], ",")[[1]])
        remainder = c(remainder, paste(strin[[1]][9], strin[[1]][12], sep="_"))
        matrices = c(matrices, paste(strin[[1]][1], strin[[1]][2], sep="_"))

}
data$noun = noun
data$remainder = remainder
data$matrices = matrices


# Read noun embedding bias
nounFreqs = read.csv("../../../../materials/nouns/corpus_counts/german/output/counts.tsv", sep="\t") %>% rename(noun=Noun)
nounFreqs = nounFreqs %>% mutate(True_False_False = log(CountWithThat))
nounFreqs = nounFreqs %>% mutate(False_False_False = log(CountWithoutThat))

data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)

data$MissingVerb = (data$verbs < 3)

library(ggplot2)
plot = ggplot(data %>% group_by(workerid) %>% summarise(MissingVerb = mean(MissingVerb, na.rm=TRUE)), aes(x=MissingVerb)) + geom_histogram() + xlab("Rate of responses with a verb missing") + ylab("Number of Subjects") + theme_bw()
ggsave(plot, file="figures/errorRates_byParticipant.pdf", height=3, width=3)



data$True_Minus_False = data$True_False_False - data$False_False_False
data$True_Minus_False.C = data$True_Minus_False - mean(data$True_Minus_False, na.rm=TRUE)



#summary(glmer(MissingVerb ~ True_Minus_False + (1|workerid) + (1|noun) + (1|remainder), data=data, family="binomial"))
#summary(glmer(MissingVerb ~ True_Minus_False + (1+True_Minus_False|workerid) + (1|noun) + (1+True_Minus_False|remainder), data=data, family="binomial"))

# https://ggplot2.tidyverse.org/reference/geom_smooth.html
#binomial_smooth <- function(...) {
#  geom_smooth(method = "glm", method.args = list(family = "binomial"), ...)
#}
library(ggplot2)
library(ggrepel)
#plot = ggplot(data %>% group_by(noun, True_Minus_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=True_Minus_False, y=MissingVerb+0.0)) + geom_point()+ geom_text_repel(aes(label=noun)) + binomial_smooth(data=data)  + theme_bw() + xlab("Embedding Bias") + ylab("Responses with Missing Verb") + geom_text(aes(label=c("beta=-0.33 (95% CrI [-0.62, -0.08])"), x=c(-1), y=c(0.5)))
#
#
#
#model_fit = data.frame(True_Minus_False = 5.5*(((1:100)/100)-1))
#model_fit$MissingVerb = 1/(1+exp(-(-2.15 + -0.33 * (model_fit$True_Minus_False))))
#model_fit$MissingVerbUpper = 1/(1+exp(-(-2.15 + -0.62 * (model_fit$True_Minus_False))))
#model_fit$MissingVerbLower = 1/(1+exp(-(-2.15 + -0.08 * (model_fit$True_Minus_False))))
#
#
#plot = ggplot(data %>% group_by(noun, True_Minus_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=True_Minus_False, y=MissingVerb+0.0)) + geom_point()+ geom_text_repel(aes(label=noun)) +  theme_bw() + xlab("Embedding Bias") + ylab("Responses with Missing Verb") + geom_line(data=model_fit) + geom_ribbon(data=model_fit, aes(ymin=MissingVerbUpper,ymax=MissingVerbLower),alpha=0.3)
#
#


# Plot by-noun rates of complete responses
plot = ggplot(data %>% group_by(noun, True_Minus_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=True_Minus_False, y=MissingVerb+0.0)) + geom_point()+ geom_text_repel(aes(label=noun)) + geom_smooth(method="lm")  + theme_bw() + xlab("Embedding Bias") + ylab("Responses with Missing Verb") + ylim(0.1, 0.8)
ggsave(plot, file="figures/rates_by_conditional.pdf", height=3, width=3)
#plot = ggplot(data %>% group_by(noun, True_False_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=True_False_False, y=MissingVerb)) + geom_point() + geom_smooth(method="lm") + geom_label(aes(label=noun))
#plot = ggplot(data %>% group_by(noun, False_False_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=False_False_False, y=MissingVerb)) + geom_point() + geom_smooth(method="lm") + geom_label(aes(label=noun))


library(brms)
model = (brm(MissingVerb ~ True_Minus_False.C + (1+True_Minus_False.C|workerid) + (1|noun) + (1+True_Minus_False.C|remainder) + (1+True_Minus_False.C|matrices), data=data, family="bernoulli"))

sink("output/analysis_replication.R.txt")
print(summary(model))
sink()

write.table(summary(model)$fixed, file="output/analysis_replication.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$Intercept = samples$b_Intercept
samples$EmbeddingBias = samples$b_True_Minus_False.C
plot = mcmc_areas(samples, pars=c("Intercept", "EmbeddingBias"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms.pdf", width=5, height=3)



samples = posterior_samples(model)
sink("output/analysis_replication.R.txt")
print(summary(model))
print(mean(samples$b_True_Minus_False.C>0))
sink()


#
#
#u = (coef(glmer(MissingVerb ~ (1|noun) + (1|workerid) + (1|remainder), data=data, family="binomial"))$noun)
#u$noun = rownames(u)
#u$Slope = u[["(Intercept)"]]
#
#u = merge(u, nounFreqs, by=c("noun"))
#
#
#plot = ggplot(u, aes(x=True_False_False-False_False_False, y=Slope)) + geom_point() + geom_smooth(method="lm") + geom_label(aes(label=noun))
#
#
#
