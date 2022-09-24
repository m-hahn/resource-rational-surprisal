
library(tidyr)
library(dplyr)
library(lme4)

data = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/collectResults_Stims.py_Staub_2016.tsv", sep="\t")

configs = unique(data %>% select(deletion_rate, predictability_weight))
#for(i in 1:nrow(configs)) {
#   model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|Item), data=data %>% filter(deletion_rate==configs$deletion_rate[[i]], predictability_weight==configs$predictability_weight[[i]]) %>% group_by(pp_rc, emb_c, someIntervention, Item, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) ))
#   print(configs[i,])
#   print(coef(summary(model)))
#}
##model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + ( 1+ pp_rc + emb_c + someIntervention|Model), data=data %>% filter()) 


library(ggplot2)

#
## Limitation: The results are confounded with sentence position, which has a strong impact on GPT2 Surprisal
#plot = ggplot(data=data %>% group_by(Script, intervention, embedding, deletion_rate, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=intervention, y=SurprisalReweighted, group=paste(Script,embedding), linetype=Script, color=embedding)) + geom_line() + facet_grid(predictability_weight ~ deletion_rate)
#ggsave(plot, file="cunnings-sturt_vanillaLSTM.pdf", height=10, width=10)
#
#

surprisalSmoothed = data.frame()

# NO SMOOTHING
for(i in 1:nrow(configs)) {
   delta = configs$deletion_rate[[i]]
   lambda = configs$predictability_weight[[i]]
   surprisals = data %>% filter(abs(deletion_rate-delta)<=0.0, abs(predictability_weight-lambda)<=0.0) %>% group_by(Region, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% mutate(deletion_rate=delta, predictability_weight=lambda)
   surprisalSmoothed = rbind(surprisalSmoothed, as.data.frame(surprisals))
}



#for(i in 1:nrow(configs)) {
#   delta = configs$deletion_rate[[i]]
#   lambda = configs$predictability_weight[[i]]
#   surprisals = data %>% filter(abs(deletion_rate-delta)<=0.05, abs(predictability_weight-lambda)<=0.25) %>% group_by(Region, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% mutate(deletion_rate=delta, predictability_weight=lambda)
#   surprisalSmoothed = rbind(surprisalSmoothed, as.data.frame(surprisals))
#}
#
                                         
surprisalSmoothed = surprisalSmoothed %>% mutate("RCType" = ifelse(Condition %in% c("A", "B", "C", "D"), "ORC", "SRC")) 
surprisalSmoothed = surprisalSmoothed %>% mutate("HasPP" = ifelse(Condition %in% c("B", "D", "F"), TRUE, FALSE)) 
surprisalSmoothed = surprisalSmoothed %>% mutate("HasParticle" = ifelse(Condition %in% c("C", "D"), TRUE, FALSE))
surprisalSmoothed = surprisalSmoothed %>% mutate(ORC.C = (RCType == "ORC")-0.5)
surprisalSmoothed = surprisalSmoothed %>% mutate(Group = ifelse(HasParticle, "ORCPhrasal", ifelse(RCType == "ORC", "ORC", "SRC")), Length = ifelse(HasPP, "Long", "Short"))  

plot = ggplot(data=surprisalSmoothed, aes(x=Region, y=SurprisalReweighted, group=Condition, color=RCType, linetype=paste(HasPP,HasParticle))) + geom_line() + facet_grid(predictability_weight ~ deletion_rate) + theme_bw()
ggsave(plot, file="figures/staub_2016_vanillaLSTM_smoothed.pdf", height=10, width=10)


# matrix verb
surprisalSmoothed$ConditionGroup = as.character(as.numeric(as.factor(surprisalSmoothed$Group)))
plot = ggplot(data=surprisalSmoothed %>% filter(Region == "V1"), aes(x=ConditionGroup, y=SurprisalReweighted, group=Length, fill=Length, color=Length)) + geom_bar(stat="identity", width=.5, position = "dodge") + facet_grid(predictability_weight ~ deletion_rate) + theme_bw()
#plot = ggplot(data=surprisalSmoothed %>% filter(Region == "V1"), aes(x=Group, y=SurprisalReweighted, group=Length, fill=Length, color=Length)) + geom_point() + facet_grid(predictability_weight ~ deletion_rate) + theme_bw()
ggsave(plot, file="figures/staub_2016_vanillaLSTM_smoothed_MatrixVerb.pdf", height=5, width=10)

plot = ggplot(data=surprisalSmoothed %>% filter(predictability_weight==1, deletion_rate >= 0.2 , deletion_rate < 0.8) %>% group_by(deletion_rate, Length, HasParticle, Region, RCType, Condition, HasPP, Group) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% filter(Region == "V1") %>% group_by(Length, HasParticle, Region, RCType, Condition, HasPP, Group) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Group, y=SurprisalReweighted, group=Length, fill=Length, color=Length)) + geom_bar(stat="identity", width=.5, position = "dodge") + theme_bw()
ggsave(plot, file="figures/staub_2016_vanillaLSTM_smoothed_MatrixVerb_selected.pdf", height=4, width=4)


plot = ggplot(data=surprisalSmoothed %>% filter(Region != "V1", !HasParticle), aes(x=Region, y=SurprisalReweighted, group=Condition, color=RCType, linetype=HasPP)) + geom_line() + facet_grid(predictability_weight ~ deletion_rate) + theme_bw()
ggsave(plot, file="figures/staub_2016_vanillaLSTM_smoothed_Inside.pdf", height=5, width=10)

plot = ggplot(data=surprisalSmoothed %>% filter(predictability_weight==1, deletion_rate >= 0.2 , deletion_rate < 0.8) %>% group_by(HasParticle, Region, RCType, Condition, HasPP) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% filter(Region != "V1", !HasParticle), aes(x=Region, y=SurprisalReweighted, group=Condition, color=RCType, linetype=HasPP)) + geom_line() + theme_bw()
ggsave(plot, file="figures/staub_2016_vanillaLSTM_smoothed_Inside_selected.pdf", height=3, width=4)




human = read.csv("human-staub_2016.tsv", sep="\t")

human = human %>% mutate("RCType" = ifelse(Condition %in% c("A", "B", "C", "D"), "ORC", "SRC")) 
human = human %>% mutate("HasPP" = ifelse(Condition %in% c("B", "D", "F"), TRUE, FALSE)) 
human = human %>% mutate("HasParticle" = ifelse(Condition %in% c("C", "D"), TRUE, FALSE))
human = human %>% mutate(ORC.C = (RCType == "ORC")-0.5)
human = human %>% mutate(Group = ifelse(HasParticle, "ORCPhrasal", ifelse(RCType == "ORC", "ORC", "SRC")), Length = ifelse(HasPP, "Long", "Short"))  

plot = ggplot(data=human %>% filter(Region != "V1", !HasPP, !HasParticle), aes(x=Region, y=FirstPass, group=Condition, color=RCType)) + geom_line() + theme_bw()
ggsave(plot, file="figures/staub_2016_FirstPass_Inner.pdf", height=3, width=7)
plot = ggplot(data=human %>% filter(Region != "V1", !HasPP, !HasParticle), aes(x=Region, y=GoPast, group=Condition, color=RCType)) + geom_line() + theme_bw()
ggsave(plot, file="figures/staub_2016_GoPast_Inner.pdf", height=3, width=7)

humanGoPast = human %>% rename(RT=GoPast) %>% mutate(Measure="ET: GoPast", FirstPass=NULL)
humanFirstPass = human %>% rename(RT=FirstPass) %>% mutate(Measure="ET: FirstPass", GoPast=NULL)
humanLong = rbind(humanGoPast, humanFirstPass)
plot = ggplot(data=humanLong %>% filter(Region != "V1", !HasPP, !HasParticle), aes(x=Region, y=RT, group=Condition, color=RCType)) + geom_line() + theme_bw() + facet_grid(~Measure)
ggsave(plot, file="figures/staub_2016_Joint_Inner.pdf", height=3, width=7)


plot = ggplot(data=human %>% filter(Region == "V1"), aes(x=Group, y=FirstPass, group=Length, fill=Length, color=Length)) + geom_bar(stat="identity", width=.5, position = "dodge") + theme_bw()
ggsave(plot, file="figures/staub_2016_FirstPass_MatrixVerb.pdf", height=3, width=3)


plot = ggplot(data=human %>% filter(Region == "V1"), aes(x=Group, y=GoPast, group=Length, fill=Length, color=Length)) + geom_bar(stat="identity", width=.5, position = "dodge") + theme_bw()
ggsave(plot, file="figures/staub_2016_GoPast_MatrixVerb.pdf", height=3, width=3)

plot = ggplot(data=humanLong %>% filter(Region == "V1"), aes(x=Group, y=RT, group=Length, fill=Length, color=Length)) + geom_bar(stat="identity", width=.5, position = "dodge") + theme_bw() + facet_grid(~Measure)
ggsave(plot, file="figures/staub_2016_Joint_MatrixVerb.pdf", height=3, width=6)


# but these come from a different stimulus set
human = read.csv("human-staub_2010.tsv", sep="\t")

human = human %>% mutate("RCType" = ifelse(Condition %in% c("A", "B", "C", "D"), "ORC", "SRC")) 
human = human %>% mutate("HasPP" = ifelse(Condition %in% c("B", "D", "F"), TRUE, FALSE)) 
human = human %>% mutate("HasParticle" = ifelse(Condition %in% c("C", "D"), TRUE, FALSE))
human = human %>% mutate(ORC.C = (RCType == "ORC")-0.5)
human = human %>% mutate(Group = ifelse(HasParticle, "ORCPhrasal", ifelse(RCType == "ORC", "ORC", "SRC")), Length = ifelse(HasPP, "Long", "Short"))  

plot = ggplot(data=human %>% filter(Region != "V1", !HasPP, !HasParticle), aes(x=Region, y=FirstPass, group=Condition, color=RCType)) + geom_line() + theme_bw()
ggsave(plot, file="figures/staub_2010_FirstPass_Inner.pdf", height=3, width=7)
plot = ggplot(data=human %>% filter(Region != "V1", !HasPP, !HasParticle), aes(x=Region, y=GoPast, group=Condition, color=RCType)) + geom_line() + theme_bw()
ggsave(plot, file="figures/staub_2010_GoPast_Inner.pdf", height=3, width=7)


