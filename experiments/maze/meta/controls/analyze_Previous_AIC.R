library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data1 = read.csv("../../previous/study5_replication/Submiterator-master/all_trials.tsv", sep="\t") %>% mutate(distractor=NA, group=NULL)
data2 = read.csv("../../experiment1/Submiterator-master/trials_byWord.tsv", sep="\t") %>% mutate(workerid=workerid+1000)
data3 = read.csv("../../experiment2/Submiterator-master/trials-experiment2.tsv", sep="\t") %>% mutate(workerid=workerid+2000)

data = rbind(data1, data2, data3)

# Exclude participants with excessive error rates
participantsByErrorsBySlide = data %>% filter(correct != "none") %>% group_by(workerid) %>% summarise(ErrorsBySlide = mean(correct == "no"))
data = merge(data, participantsByErrorsBySlide, by=c("workerid"))
data = data %>% filter(ErrorsBySlide < 0.2)

# Only consider critical trials
data = data %>% filter(condition != "filler")

# Remove trials with incorrect responses
data = data %>% filter(rt > 0, correct == "yes")

# Remove extremely low or extremely high reading times
data = data %>% filter(rt < quantile(data$rt, 0.99))
data = data %>% filter(rt > quantile(data$rt, 0.01))


# Load corpus counts (COCA)
nounsCOCA = read.csv("../../../../materials/nouns/corpus_counts/COCA/results/results_counts4.py.tsv", sep="\t")                                                                                                            
nounsCOCA$Conditional_COCA = log(nounsCOCA$theNOUNthat/nounsCOCA$theNOUN)                                                                                                                                   
nounsCOCA$Marginal_COCA = log(nounsCOCA$theNOUN)
nounsCOCA$Joint_COCA = log(nounsCOCA$theNOUNthat)

data$Noun = data$noun

data = merge(data, nounsCOCA, by=c("Noun"))


data$Conditional_COCA.C = data$Conditional_COCA - mean(data$Conditional_COCA, na.rm=TRUE)
data$Marginal_COCA.C = data$Marginal_COCA - mean(data$Marginal_COCA, na.rm=TRUE)
data$Joint_COCA.C = data$Joint_COCA - mean(data$Joint_COCA, na.rm=TRUE)


# Load corpus counts (ukWac)
nounsukwac = read.csv("../../../../materials/nouns/corpus_counts/ukwac/results/results_counts4.py.tsv", sep="\t")                                                                                                          
nounsukwac$Conditional_ukwac = log(nounsukwac$theNOUNthat/nounsukwac$theNOUN)                                                                                                                               
nounsukwac$Marginal_ukwac = log(nounsukwac$theNOUN)
nounsukwac$Joint_ukwac = log(nounsukwac$theNOUNthat)

data = merge(data, nounsukwac, by=c("Noun"))
 
data$Conditional_ukwac.C = data$Conditional_ukwac - mean(data$Conditional_ukwac, na.rm=TRUE)
data$Marginal_ukwac.C = data$Marginal_ukwac - mean(data$Marginal_ukwac, na.rm=TRUE)
data$Joint_ukwac.C = data$Joint_ukwac - mean(data$Joint_ukwac, na.rm=TRUE)

# Load corpus counts (Wikipedia)
nounFreqs = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/results_counts4NEW.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)

nounFreqs2 = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs, by=c("noun"), all.x=TRUE)

# Load SC Bias
scrc = read.csv("../../../../materials/nouns/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t") %>% rename(noun=Noun)
scrc = scrc %>% mutate(SC_Bias = (SC+1)/(SC+RC+2))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

data = merge(scrc, data, by=c("noun"), all=TRUE)

# Code Embedding Bias
data = data %>% mutate(EmbeddingBias = True_False_False-False_False_False)
data = data %>% mutate(EmbeddingBias.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))
data = data %>% mutate(JointNounThat = True_False_False)
data = data %>% mutate(MarginalNoun = False_False_False)
data = data %>% mutate(JointNounThat.C = JointNounThat-mean(JointNounThat, na.rm=TRUE))
data = data %>% mutate(MarginalNoun.C = MarginalNoun-mean(MarginalNoun, na.rm=TRUE))



# Code Conditions
data = data %>% mutate(compatible = ifelse(condition %in% c("critical_SCRC_compatible", "critical_compatible"), TRUE, FALSE))
data = data %>% mutate(HasSC = ifelse(condition == "critical_NoSC", FALSE, TRUE))
data = data %>% mutate(HasRC = ifelse(condition %in% c("critical_SCRC_compatible", "critical_SCRC_incompatible"), TRUE, FALSE))

# Contrast Coding
data$HasRC.C = ifelse(data$HasRC, 0.5, ifelse(data$HasSC, -0.5, 0)) #resid(lm(HasRC ~ HasSC, data=data))
data$HasSC.C = data$HasSC - 0.8
data$compatible.C = ifelse(!data$HasSC, 0, ifelse(data$compatible, 0.5, -0.5))
# Log-Transform Reading Times
data$LogRT = log(data$rt)

# Center trial order
data$trial = data$trial - mean(data$trial, na.rm=TRUE)











library(lme4)






subjects = read.csv("../../../../materials/nouns//corpus_counts/wikipedia/results/results_counts_Parsed.py.tsv" , sep="\t") %>% mutate(LogSubjectBias = log(Subject+0.000001) - log(Bare+0.000001))
data = merge(data, subjects, by=c("Noun"), all.x=TRUE)
data$LogSubjectBias.C = data$LogSubjectBias - mean(data$LogSubjectBias, na.rm=TRUE)



verbal = read.csv("../../../../../noisy-channel-structural-forgetting/nouns/English/nouns_verbal.tsv", sep="\t")
data = merge(data, verbal, by=c("Noun"), all.x=TRUE)
data$HomophoneVerb.C = data$HomophoneVerb - mean(data$HomophoneVerb, na.rm=TRUE)
data$Deverbal.C = data$Deverbal - mean(data$Deverbal, na.rm=TRUE)

library(stringr)
data$NounLength = str_length(data$Noun)
data$NounLength.C = data$NounLength - mean(data$NounLength, na.rm=TRUE)

data$Log_SC_Bias.C = data$Log_SC_Bias - mean(data$Log_SC_Bias, na.rm=TRUE)

data = data %>% filter(!is.na(EmbeddingBias.C), !is.na(Deverbal.C), !is.na(Log_SC_Bias.C), !is.na(LogSubjectBias.C), !is.na(Conditional_ukwac.C), !is.na(Conditional_COCA.C))


formula = "LogRT ~ trial + HasRC.C * compatible.C + (1|noun) + (1|workerid) + (1|item)"
bestBIC = 1000000000000000000
predictors = c("EmbeddingBias.C", "JointNounThat.C", "MarginalNoun.C", "Log_SC_Bias.C", "HomophoneVerb.C", "Deverbal.C", "LogSubjectBias.C", "Conditional_ukwac.C", "Marginal_ukwac.C", "Joint_ukwac.C", "Conditional_COCA.C", "Marginal_COCA.C", "Joint_COCA.C", "NounLength.C")
for(iter in (1:50)) {
        bestFormulaHere = "NONE"
	bestBICHere = bestBIC
	for(pred in predictors) {
		formula2 = paste(formula, "+", pred)
		bic = BIC(lmer(formula2, data=data %>% filter(HasSC.C>0, Region=="REGION_3_0"), REML=FALSE))
		cat(formula2, "\t", bic, "\n")
		if(bic < bestBICHere) {
			bestBICHere = bic
			bestFormulaHere=formula2
		}
	}
	if(bestFormulaHere!="NONE") {
		cat(bestFormulaHere, "\n")
		formula = bestFormulaHere
		bestBIC=bestBICHere
	} else {
		break
	}
}


sink("output/analyze_Previous_BIC.R.txt")
cat("BEST MODEL", formula)
cat("\n")
cat("BIC", bestBIC)
cat("\n")
sink()


