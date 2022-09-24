library(tidyr)
library(dplyr)
library(lme4)


# Read trial data
data = read.csv("trials.tsv", sep="\t")

# Read the annotated completions
annotated = read.csv("annotated.tsv", sep="\t")
data = merge(data, annotated, by=c("completion"), all=TRUE)

# Select critical trials
data = data %>% filter(condition == "SC_RC")

library(stringr)

noun = c()
remainder = c()
sentences = as.character(data$sentence1)
for(i in (1:nrow(data))) {
        strin = strsplit(sentences[i], " ")
        noun = c(noun, strin[[1]][2])
        remainder = c(remainder, paste(strin[[1]][5], strin[[1]][6], sep="_"))
}
data$noun = noun
data$remainder = remainder


# Read noun embedding bias
nounFreqs = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)


nounFreqs2 = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)


nounsCOCA = read.csv("../../../../materials/nouns/corpus_counts/COCA/results/results_counts4.py.tsv", sep="\t")                                                                                           
nounsCOCA$Conditional_COCA = log(nounsCOCA$theNOUNthat/nounsCOCA$theNOUN)
nounsCOCA$Marginal_COCA = log(nounsCOCA$theNOUN)
nounsCOCA$Joint_COCA = log(nounsCOCA$theNOUNthat)

data = merge(data, nounsCOCA %>% rename(noun=Noun), by=c("noun"))

data$Conditional_COCA.C = data$Conditional_COCA - mean(data$Conditional_COCA, na.rm=TRUE)


data$MissingVerb = (data$verbs < 3)

library(ggplot2)
#plot = ggplot(data %>% group_by(workerid) %>% summarise(MissingVerb = mean(MissingVerb, na.rm=TRUE)), aes(x=MissingVerb)) + geom_histogram() + xlab("Rate of responses with a verb missing") + ylab("Number of Subjects") + theme_bw()
#ggsave(plot, file="figures/errorRates_byParticipant.pdf", height=3, width=3)
#


data$True_Minus_False = data$True_False_False - data$False_False_False
data$True_Minus_False.C = data$True_Minus_False - mean(data$True_Minus_False, na.rm=TRUE)


library(ggplot2)
library(ggrepel)

# Plot by-noun rates of complete responses
plot = ggplot(data %>% group_by(noun, Conditional_COCA) %>% summarise(MissingVerb=mean(MissingVerb, na.rm=TRUE)), aes(x=Conditional_COCA, y=MissingVerb+0.0)) + geom_point()+ geom_text_repel(aes(label=noun)) + geom_smooth(method="lm")  + theme_bw() + xlab("Embedding Bias") + ylab("Responses with Missing Verb") + ylim(0.1, 0.8)
ggsave(plot, file="figures/rates_by_conditional_COCA.pdf", height=3, width=3)

library(brms)
model = (brm(MissingVerb ~ Conditional_COCA.C + (1+Conditional_COCA.C|workerid) + (1|noun) + (1+Conditional_COCA.C|remainder), data=data, family="bernoulli"))


write.table(summary(model)$fixed, file="output/analysis_Expt3_en_COCA.R_fixed.tsv", sep="\t")

library(bayesplot)

samples = posterior_samples(model)

samples$Intercept = samples$b_Intercept
samples$EmbeddingBias_COCA = samples$b_Conditional_COCA.C
plot = mcmc_areas(samples, pars=c("Intercept", "EmbeddingBias_COCA"), prob=.95, n_dens=32, adjust=5)
ggsave(plot, file="figures/posterior-histograms_COCA.pdf", width=5, height=3)



sink("output/analysis_replication_COCA.R.txt")
print(summary(model))
print(mean(samples$b_Conditional_COCA.C>0))
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
