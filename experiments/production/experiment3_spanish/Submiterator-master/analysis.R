library(tidyr)
library(dplyr)
library(lme4)


# Read trial data
data = read.csv("all-trials.tsv", sep="\t")
data$verbs=NULL

# Read annotation of completions
annotated = read.csv("annotated_Medium.tsv", sep="\t")
data = merge(data, annotated, by=c("completion"), all=FALSE)

# Select critical trials
data = data %>% filter(condition == "SC_RC")

library(stringr)

noun = c()
remainder = c()
sentences = as.character(data$sentence1)
for(i in (1:nrow(data))) {
        strin = strsplit(sentences[i], " ")
        noun = c(noun, strin[[1]][2])
        remainder = c(remainder, paste(strin[[1]][6], sep="_"))
}
data$noun = noun
data$remainder = remainder


# Read noun embedding bias
nounFreqs = read.csv("../../../../materials/nouns/corpus_counts/spanish/output/counts.tsv", sep="\t") %>% rename(noun=Noun)
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

library(ggplot2)
library(ggrepel)

# Plot by-noun rates of complete responses
plot = ggplot(data %>% group_by(noun, True_Minus_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=True_Minus_False, y=MissingVerb)) + geom_point() + geom_smooth(method="lm") + geom_text_repel(aes(label=noun)) + theme_bw() + xlab("Embedding Bias") + ylab("Responses with Verb Missing") + ylim(0.1, 0.8)
ggsave(plot, file="figures/rates_by_conditional.pdf", height=3, width=3)
#plot = ggplot(data %>% group_by(noun, True_False_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=True_False_False, y=MissingVerb)) + geom_point() + geom_smooth(method="lm") + geom_label(aes(label=noun))
#plot = ggplot(data %>% group_by(noun, False_False_False) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=False_False_False, y=MissingVerb)) + geom_point() + geom_smooth(method="lm") + geom_label(aes(label=noun))


library(brms)
model = (brm(MissingVerb ~ True_Minus_False.C + (1+True_Minus_False.C|workerid) + (1|noun) + (1+True_Minus_False.C|remainder), data=data, family="bernoulli"))

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
