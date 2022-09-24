library(tidyr)
library(dplyr)
data = read.csv("/u/scr/mhahn/reinforce-logs-both-short/full-logs-retentionProbs/SUMMARY_collectRetentionRates_p.py.tsv", sep=" ", header=F, col.names = c("Word", "POS", "Position", "Rate", "ID", "lambda", "delta", "WordFreq"))

data$LogWordFreq = log(data$WordFreq)

library(ggplot2)

ud = read.csv("en-ptb-map.tsv", sep="\t", header=F, col.names=c("POS", "UD_POS"))
data = merge(data, ud, by=c("POS"), all.x=TRUE)

data$UD_POS = as.character(data$UD_POS)
data$UD_POS = ifelse(data$Word == "that", "that", data$UD_POS)



plot = ggplot(data %>% mutate(delta=20*(1-delta)) %>% filter(lambda==1) %>% group_by(delta) %>% group_by(LogWordFreq, delta, Word, Position) %>% summarise(Rate=mean(Rate)), aes(x=Position, y=Rate, group=Word, color=LogWordFreq)) + geom_line(se=F, alpha=0.5) + facet_wrap(~delta) + theme_bw() + scale_x_reverse() + xlim(-15, 1) + ylim(0, 1.1)
ggsave(plot, file="figures/retention_rates_lambda1_20_raw_overall.pdf", height=4, width=6)



plot = ggplot(data %>% mutate(delta=20*(1-delta)) %>% filter(lambda==1) %>% group_by(delta) %>% group_by(LogWordFreq, delta, Word, Position) %>% summarise(Rate=mean(Rate)) %>% filter(delta>=5, delta<=16) %>% group_by(LogWordFreq, Word, Position)  %>% summarise(Rate=mean(Rate, na.rm=TRUE)), aes(x=Position, y=Rate, group=Word, color=LogWordFreq)) + geom_line(se=F, alpha=0.5) + theme_bw() + scale_x_reverse() + xlim(-15, 1) + ylim(0, 1.1)
ggsave(plot, file="figures/retention_rates_lambda1_20_raw_overall_average.pdf", height=3, width=3)



u = data %>% mutate(delta=20*(1-delta)) %>% filter(lambda==1) %>% group_by(delta) %>% filter(UD_POS %in% c("that", "ADP")) %>% group_by(Word, UD_POS, delta, Position) %>% summarise(Rate=mean(Rate))
plot = ggplot(u, aes(x=Position, y=Rate, group=Word, color=UD_POS)) 
plot = plot + geom_line(data=u %>% filter(UD_POS == "ADP"), alpha=0.3)
plot = plot + geom_line(data=u %>% filter(UD_POS == "that"), alpha=1)
plot = plot + facet_wrap(~delta) + theme_bw() + scale_x_reverse() + xlim(-10, 1) + ylim(0,1.1) + theme(axis.text=element_text(size=5))
ggsave(plot, file="figures/retention_rates_lambda1_20_raw_overall_functionWords.pdf", height=4, width=6)


u = data %>% mutate(delta=20*(1-delta)) %>% filter(lambda==1) %>% group_by(delta) %>% filter(UD_POS %in% c("that", "ADP")) %>% group_by(Word, UD_POS, delta, Position) %>% summarise(Rate=mean(Rate)) %>% group_by(Word, UD_POS, Position) %>% summarise(Rate=mean(Rate))
plot = ggplot(u, aes(x=Position, y=Rate, group=Word, color=UD_POS)) 
plot = plot + geom_line(data=u %>% filter(UD_POS == "ADP"), alpha=0.3)
plot = plot + geom_line(data=u %>% filter(UD_POS == "that"), alpha=1)
plot = plot + theme_bw() + scale_x_reverse() + xlim(-10, 1) + ylim(0,1.1)
ggsave(plot, file="figures/retention_rates_lambda1_20_raw_overall_functionWords_average.pdf", height=3, width=4)






sink("figures/collectRetentionRates_p_analyze.R.tsv")
cat("delta", "position", "t_position", "logWordFreq", "t_logWordFreq", "interaction", "t_interaction", sep="\t")
for(delta in (1:19)) {
  delta_ = 1-(delta/20)
  results = coef(summary(lm(Rate ~ Position*LogWordFreq, data= data %>% filter(delta==delta_, lambda==1) %>% group_by(Word, UD_POS, POS, Position, LogWordFreq) %>% summarise(Rate=mean(Rate)))) )
  position = results[2,1]
  logWordFreq = results[3,1]
  interaction = results[4,1]
  t_position = results[2,4]
  t_logWordFreq = results[3,4]
  t_interaction = results[4,4]
  cat(delta, position, t_position, logWordFreq, t_logWordFreq, interaction, t_interaction, sep="\t")
  cat("\n")
}
sink()

