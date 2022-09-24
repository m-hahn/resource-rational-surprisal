
library(ggplot2)
library(dplyr)
library(tidyr)
model = read.csv("../../../model/compute_surprisal/analyze_output/prepareMeansByExperiment_E1_ByStimuli.R.tsv", sep="\t")
nounsRatingsL = (model %>% group_by(Noun) %>% summarise(count = NROW(Condition)) %>% filter(count > 1000))$Noun
nounsRatingsL = unique(nounsRatingsL)
nounsMaze = as.character(unique(read.csv("../../../experiments/maze/experiment2/Submiterator-master/trials-experiment2.tsv", sep="\t")$noun))
nouns = unique(c(as.character(nounsRatingsL), as.character(nounsMaze)))
nouns = nouns[!is.na(nouns)]
nouns_counts = read.csv("../corpus_counts/COCA/results/results_counts4.py.tsv", sep="\t")
# this should be all TRUE
print(nouns %in% nouns_counts$Noun)
nouns = nouns_counts[nouns_counts$Noun %in% nouns,]

nouns$Conditional = log(nouns$theNOUNthat)-log(nouns$theNOUN)

CILower = c()
CIUpper = c()

for(i in 1:nrow(nouns)) {

   joint = round((nouns$theNOUNthat[[i]]))
   unigram = round((nouns$theNOUN[[i]]))
   if(unigram > 0) {
      CILower = c(CILower, (binom.test(joint, unigram)$conf.int[[1]]))
      CIUpper = c(CIUpper, (binom.test(joint, unigram)$conf.int[[2]]))
   } else {
      CILower = c(CILower, 0.0)
      CIUpper = c(CIUpper, 1.0)
   }

}

nouns$CILower = log(CILower)
nouns$CIUpper = log(CIUpper)

nouns$Conditional = log(nouns$theNOUNthat) - log(nouns$theNOUN)
types = read.csv("nounsTypes.tsv", sep="\t")
nouns = merge(nouns, types, by=c("Noun"), all.x=TRUE)
nouns$Noun = factor(nouns$Noun, levels=nouns$Noun[order(nouns$Conditional)])

nouns$InMaze = ifelse((nouns$Noun %in% nounsMaze), "bold", "plain")

library(ggrepel)
plot = ggplot(nouns, aes(x=Conditional, y=Noun, color=Group, group=Group)) + geom_point() + geom_errorbarh(aes(xmax=CIUpper, xmin=CILower, color=Group)) + geom_text_repel(aes(label=Noun, fontface=InMaze)) + theme_bw() + theme(axis.text.x=element_text(angle = 90)) + xlab("Embedding Bias (COCA)") + ylab(NULL)
ggsave(plot, file="figures/All_nouns_byType_COCA.pdf", height=12, width=6)


nouns[order(nouns$Conditional),]



