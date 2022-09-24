library(tidyr)
library(dplyr)


library(ggplot2)

data = data.frame(Ratio = c(-5, 0), Incorrect = c(0.65, 0.35))


plot = ggplot(data, aes(x=Ratio, y=Incorrect)) + geom_smooth(method="lm", se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Responses with Verb Missing")  + theme(axis.text = element_blank(), axis.ticks = element_blank()) + ylim(0,1)
ggsave(plot, file="figures/conceptual-predictions-production.pdf", width=2, height=2)




