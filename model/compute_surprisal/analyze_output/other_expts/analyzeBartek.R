
library(tidyr)
library(dplyr)
library(lme4)

data = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/collectResults_Stims.py_BartekEtal.tsv", sep="\t")

data$Embedding = NA
data$Intervening = NA

data[data$Condition == "a",]$Embedding = "Matrix"
data[data$Condition == "a",]$Intervening = "none"

data[data$Condition == "b",]$Embedding = "Matrix"
data[data$Condition == "b",]$Intervening = "pp"

data[data$Condition == "c",]$Embedding = "Matrix"
data[data$Condition == "c",]$Intervening = "rc"

data[data$Condition == "d",]$Embedding = "Embedded"
data[data$Condition == "d",]$Intervening = "none"

data[data$Condition == "e",]$Embedding = "Embedded"
data[data$Condition == "e",]$Intervening = "pp"

data[data$Condition == "f",]$Embedding = "Embedded"
data[data$Condition == "f",]$Intervening = "rc"


data = data %>% mutate(pp_rc = case_when(Intervening == "rc" ~ 1, Intervening == "pp" ~ -1, TRUE ~ 0))
data = data %>% mutate(emb_c = case_when(Embedding == "Matrix" ~ -1, Embedding == "Embedded" ~ 1))
data = data %>% mutate(someIntervention = case_when(Intervening == "none" ~ -1, TRUE ~ 1))

configs = unique(data %>% select(deletion_rate, predictability_weight))
#for(i in 1:nrow(configs)) {
#   model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|Item), data=data %>% filter(deletion_rate==configs$deletion_rate[[i]], predictability_weight==configs$predictability_weight[[i]]) %>% group_by(pp_rc, emb_c, someIntervention, Item, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) ))
#   print(configs[i,])
#   print(coef(summary(model)))
#}
#model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + ( 1+ pp_rc + emb_c + someIntervention|Model), data=data %>% filter()) 


library(ggplot2)


# Limitation: The results are confounded with sentence position, which has a strong impact on GPT2 Surprisal
plot = ggplot(data=data %>% group_by(Script, Intervening, Embedding, deletion_rate, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Intervening, y=SurprisalReweighted, group=paste(Script,Embedding), linetype=Script, color=Embedding)) + geom_line() + facet_grid(predictability_weight ~ deletion_rate)
ggsave(plot, file="figures/bartek_bb_vanillaLSTM.pdf", height=10, width=10)



surprisalSmoothed = data.frame()

for(i in 1:nrow(configs)) {
   delta = configs$deletion_rate[[i]]
   lambda = configs$predictability_weight[[i]]
 # No smoothing
#   surprisals = data %>% filter(abs(deletion_rate-delta)<=0.05, abs(predictability_weight-lambda)<=0.25) %>% group_by(Intervening, Embedding) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% mutate(deletion_rate=delta, predictability_weight=lambda)
   surprisals = data %>% filter(abs(deletion_rate-delta)<=0.0, abs(predictability_weight-lambda)<=0.0) %>% group_by(Intervening, Embedding) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% mutate(deletion_rate=delta, predictability_weight=lambda)
   surprisalSmoothed = rbind(surprisalSmoothed, as.data.frame(surprisals))
}

surprisalSmoothed[surprisalSmoothed$Intervening == "none",]$Intervening = "-"
plot = ggplot(data=surprisalSmoothed %>% group_by(Intervening, Embedding, deletion_rate, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Intervening, y=SurprisalReweighted, group=paste(Embedding), color=Embedding)) + theme_bw() + geom_line() + facet_grid(predictability_weight ~ deletion_rate) + ylab("Model Surprisal") + theme(legend.position = 'bottom')
ggsave(plot, file="figures/bartek_bb_vanillaLSTM_smoothed.pdf", height=4, width=10)


plot = ggplot(data=surprisalSmoothed %>% filter(predictability_weight == 1, (deletion_rate >= 0.2 & deletion_rate < 0.8)) %>% group_by(Intervening, Embedding, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Intervening, y=SurprisalReweighted, group=paste(Embedding), color=Embedding)) + theme_bw() + geom_line() + ylab("Model Surprisal") + theme(legend.position = 'bottom')
ggsave(plot, file="figures/bartek_bb_vanillaLSTM_smoothed_selected.pdf", height=3, width=4)


human = read.csv("analyzeBartek_human.tsv", sep="\t") %>% filter(Stimuli == "Bartek et al.")
plot = ggplot(data=human, aes(x=Intervening, y=ReadingTime, group=paste(Embedding), color=Embedding)) + theme_bw() + geom_line() + facet_grid(~Measure) + ylab("Reading Time")  + theme(legend.position = 'bottom')
ggsave(plot, file="figures/bartek_bb_human.pdf", height=3, width=7)

