library(ggplot2)
library(dplyr)
library(tidyr)


# Load trial data
data = read.csv("trials-experiment2.tsv", sep="\t")

trialsWithError = data %>% group_by(workerid, item) %>% summarise(HasError = max(correct == "no"))

participantsByErrors = trialsWithError %>% group_by(workerid) %>% summarise(Errors = mean(HasError))

participantsByErrorsBySlide = data %>% filter(correct != "none") %>% group_by(workerid) %>% summarise(ErrorsBySlide = mean(correct == "no"))

plot = ggplot(participantsByErrorsBySlide, aes(x=ErrorsBySlide)) + geom_histogram() + theme_bw() + xlab("Fraction of Trials (Slides) with Error")
ggsave(plot, file="figures/slides-errors.pdf", width=3, height=3)
sink("output/analyze_Experiment2_ErrorsStat.R.txt")
print(median(participantsByErrorsBySlide$ErrorsBySlide))
print(mean(participantsByErrorsBySlide$ErrorsBySlide>0.2))
sink()

plot = ggplot(data %>% group_by(wordInItem) %>% summarise(error_rate=mean(correct == "no")), aes(x=wordInItem, y=error_rate)) + geom_line() + xlab("Word Number") + ylab("Error Rate")
ggsave(plot, file="figures/errors-by-position.pdf", width=5, height=3)



