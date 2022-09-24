nouns = read.csv("nouns_german.tsv", sep="\t")

library(ggplot2)
library(tidyr)
library(dplyr)


CILower = c()
CIUpper = c()

for(i in 1:nrow(nouns)) {
   joint = round((nouns$Joint[[i]]))
   unigram = round((nouns$Unigram[[i]]))
   CILower = c(CILower, log(binom.test(joint, unigram)$conf.int[[1]]))
   CIUpper = c(CIUpper, log(binom.test(joint, unigram)$conf.int[[2]]))

}

nouns$CILower = CILower
nouns$CIUpper = CIUpper

nouns$Conditional = log(nouns$Joint/nouns$Unigram)
nouns$Noun = factor(nouns$Noun, levels=nouns$Noun[order(nouns$Conditional)])

plot = ggplot(nouns, aes(y=Conditional, x=Noun)) + geom_point() + geom_errorbar(aes(ymax=CIUpper, ymin=CILower)) + theme_bw() + theme(axis.text.x=element_text(angle = 90)) + ylab("log P(dass|ART NOUN)")
ggsave(plot, file="../figures/nouns_german.pdf", width=5, height=4)
