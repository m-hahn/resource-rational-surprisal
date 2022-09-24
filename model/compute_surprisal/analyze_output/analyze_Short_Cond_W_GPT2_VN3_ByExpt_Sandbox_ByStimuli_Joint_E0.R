library(tidyr)
library(dplyr)


data_E1 = read.csv("prepareMeansByExperiment_E0_ByStimuli.R.tsv", quote='"', sep="\t") %>% mutate(Experiment = "Experiment0")


data = rbind(data_E1)

counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("~/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)


library(ggplot2)


library(lme4)



data$compatible = grepl("_co", data$Condition)
data$HasSC = !grepl("NoSC", data$Condition)
data$HasRC = grepl("RC", data$Condition)

data$HasSCHasRC = (paste(data$HasSC, data$HasRC, sep="_"))


plot = ggplot(unique(data %>% select(predictability_weight, deletion_rate, ID)) %>% group_by(predictability_weight, deletion_rate) %>% summarise(ModelRuns=NROW(ID)), aes(x=0, y=0)) + geom_text(aes(label=ModelRuns)) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none")
ggsave(plot, file="figures/modelRuns_E0.pdf", height=4, width=7)

# the full raw predictions
plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 


plot = ggplot(data %>% mutate(ContinueTraining = grepl("_TPL", Script)) %>% filter(Experiment=="Experiment0", predictability_weight==1, Region == "V1_0") %>% group_by(ContinueTraining, compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(ContinueTraining~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-WithAndWithoutTP_E0.pdf", height=4, width=10)




plot = ggplot(data %>% filter(Experiment=="Experiment1", deletion_rate>=0.3, deletion_rate<=0.7, !grepl("_TPL", Script), !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-full-NoTP_E0.pdf", height=8, width=10)


plot = ggplot(data %>% filter(Experiment=="Experiment1", grepl("_TPL", Script), !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-full-TP-NoLimit_E0.pdf", height=3.5, width=10)





plot = ggplot(data %>% filter(Experiment=="Experiment1", !grepl("_TPL", Script), !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-full-NoTP-NoLimit_E0.pdf", height=3.5, width=10)

plot = ggplot(data %>% filter(Experiment=="Experiment1", !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-full-NoLimit_E0.pdf", height=3.5, width=10)

# geom_rect idea is due to user paul at
# https://stackoverflow.com/questions/58336449/highlight-draw-a-box-around-some-of-the-plots-when-using-facet-grid-in-ggplo
df = data %>% filter(Experiment=="Experiment1", !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(df, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)
plot = plot + geom_rect(data = df %>% filter(deletion_rate >= 0.2, deletion_rate<0.8, predictability_weight==1), 
                          fill = NA, colour = "gray", xmin = -Inf,xmax = Inf,
            ymin = -Inf,ymax = Inf, size=3) 
plot = plot + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-full-NoLimit_Rectangle_E0.pdf", height=3.5, width=10)










plot = ggplot(data %>% mutate(retention_rate=1-deletion_rate) %>% filter(Experiment=="Experiment1", !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, retention_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~retention_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-full-NoLimit_Retention_E0.pdf", height=3.5, width=10)




plot = ggplot(data %>% filter(Experiment=="Experiment0", deletion_rate>=0.3, deletion_rate<=0.7, !grepl("_TPL", Script), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-full-NoTP_E0.pdf", height=8, width=10)



plot = ggplot(data %>% filter(Experiment=="Experiment0", !grepl("_TPL", Script), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-full-NoTP-NoLimit_E0.pdf", height=3.5, width=10)


plot = ggplot(data %>% filter(Experiment=="Experiment0", Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-full-NoLimit_E0.pdf", height=3.5, width=10)



df = data %>% filter(Experiment=="Experiment0", Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(df, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)
plot = plot + geom_rect(data = df %>% filter(deletion_rate >= 0.2, deletion_rate<0.8, predictability_weight==1), 
                          fill = NA, colour = "gray", xmin = -Inf,xmax = Inf,
            ymin = -Inf,ymax = Inf, size=3) 
plot = plot + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-full-NoLimit_Rectangle_E0.pdf", height=3.5, width=10)







plot = ggplot(data %>% filter(Experiment=="Experiment0", grepl("_TPL", Script), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-full-TP-NoLimit_E0.pdf", height=3.5, width=10)






plot = ggplot(data %>% filter(Experiment == "Experiment1", deletion_rate>=0.3, deletion_rate<=0.7, !grepl("_TPL", Script), !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model ThatFraction") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + ylim(0,100)
ggsave(plot, file="figures/model-critical-experiment1-full-NoTP-thatFraction_E0.pdf", height=8, width=10)



plot = ggplot(data %>% filter(Experiment == "Experiment0", deletion_rate>=0.3, deletion_rate<=0.7, !grepl("_TPL", Script), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model ThatFraction") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + ylim(0,100)
ggsave(plot, file="figures/model-critical-experiment2-full-NoTP-thatFraction_E0.pdf", height=8, width=10)



plot = ggplot(data %>% filter(Experiment == "Experiment1", !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model ThatFraction") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + ylim(0,100)
ggsave(plot, file="figures/model-critical-experiment1-full-NoLimit-thatFraction_E0.pdf", height=5, width=10)



plot = ggplot(data %>% filter(Experiment == "Experiment0", Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model ThatFraction") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + ylim(0,100)
ggsave(plot, file="figures/model-critical-experiment2-full-NoLimit-thatFraction_E0.pdf", height=5, width=10)




plot = ggplot(data %>% filter(Experiment == "Experiment1", grepl("_TPL", Script), !grepl("_co", Condition), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model ThatFraction") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + ylim(0,100)
ggsave(plot, file="figures/model-critical-experiment1-thatFraction_E0.pdf", height=1.8, width=10)



plot = ggplot(data %>% filter(Experiment == "Experiment0", grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model ThatFraction") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + ylim(0,100)
ggsave(plot, file="figures/model-critical-experiment2-thatFraction_E0.pdf", height=1.8, width=10)





plot = ggplot(data %>% filter(Experiment == "Experiment1", grepl("_TPL", Script), !grepl("_co", Condition), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1_E0.pdf", height=1.8, width=10)



plot = ggplot(data %>% filter(Experiment == "Experiment1", grepl("_TPL", Script), !grepl("_co", Condition), predictability_weight==1, Region == "V1_0")  %>% mutate(deletion_rate=floor(10*deletion_rate)/10) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-ByTens_E0.pdf", height=1.8, width=5)





plot = ggplot(data %>% filter(Experiment == "Experiment1", grepl("_TPL", Script), deletion_rate==round(10*deletion_rate)/10,!grepl("_co", Condition), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-tens_E0.pdf", height=1.8, width=5)




plot = ggplot(data %>% filter(Experiment == "Experiment1", deletion_rate>=0.2, deletion_rate<0.8, grepl("_TPL", Script), !grepl("_co", Condition), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) , aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-average_E0.pdf", height=1.8, width=1.8)



data0 = data %>% filter(Experiment == "Experiment1", deletion_rate>=0.2, deletion_rate<0.8, grepl("_TPL", Script), !grepl("_co", Condition), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 7, SurprisalReweighted <=11.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-average-points_E0.pdf", height=3.5, width=1.8)




data0 = data %>% filter(Experiment == "Experiment1", deletion_rate==0.05, !grepl("_co", Condition), predictability_weight==0.25, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 7, SurprisalReweighted <=11.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-005-points_E0.pdf", height=3.5, width=1.8)




data0 = data %>% filter(Experiment == "Experiment1", deletion_rate==0.05, !grepl("_co", Condition), predictability_weight==0.25, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 7, SurprisalReweighted <=11.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-005-points_SQUARE_E0.pdf", height=1.8, width=1.8)




data0 = data %>% filter(Experiment == "Experiment1", deletion_rate==0.5, grepl("_TPL", Script), !grepl("_co", Condition), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 7, SurprisalReweighted <=11.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-05-points_E0.pdf", height=3.5, width=1.8)



data0 = data %>% filter(Experiment == "Experiment1", deletion_rate==0.55, !grepl("_co", Condition), predictability_weight==0.75, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 7, SurprisalReweighted <=11.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-075-055-points_E0.pdf", height=3.5, width=1.8)




data0 = data %>% filter(Experiment == "Experiment0", deletion_rate>=0.35, deletion_rate<=0.8, predictability_weight==0.75, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 6, SurprisalReweighted <=10.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-075-average-points_E0.pdf", height=3.5, width=1.8)




data0 = data %>% filter(Experiment == "Experiment0", deletion_rate>=0.2, deletion_rate<0.8, grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 6, SurprisalReweighted <=10.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-average-points_E0.pdf", height=3.5, width=1.8)


data0 = data %>% filter(Experiment == "Experiment0", deletion_rate==0.5, grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 5.5, SurprisalReweighted <=11.5), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-05-points_E0.pdf", height=3.5, width=1.8)



data0 = data %>% filter(Experiment == "Experiment0", predictability_weight<1, deletion_rate==0.05, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 5.5, SurprisalReweighted <=11.5), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-005-points_E0.pdf", height=3.5, width=1.8)




data0 = data %>% filter(Experiment == "Experiment0", deletion_rate==0.55, predictability_weight==0.75, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 5.5, SurprisalReweighted <=11.5), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-075-056-points_E0.pdf", height=3.5, width=1.8)





plot = ggplot(data %>% filter(Experiment == "Experiment1", !grepl("_TPL", Script), !grepl("_co", Condition), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-noTP_E0.pdf", height=1.8, width=10)




plot = ggplot(data %>% filter(Experiment == "Experiment0", grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2_E0.pdf", height=1.8, width=10)




plot = ggplot(data %>% filter(Experiment == "Experiment0", grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% mutate(deletion_rate=floor(10*deletion_rate)/10) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-byTens_E0.pdf", height=1.8, width=5)





plot = ggplot(data %>% filter(Experiment == "Experiment0", grepl("_TPL", Script), deletion_rate==round(10*deletion_rate)/10, predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-tens_E0.pdf", height=1.8, width=5)


plot = ggplot(data %>% filter(Experiment == "Experiment0", !grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-noTP_E0.pdf", height=1.8, width=5)



plot = ggplot(data %>% filter(Experiment == "Experiment0", deletion_rate>=0.2, deletion_rate<0.8, grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) , aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-average_E0.pdf", height=1.8, width=1.8)




   dataSmoothed = data.frame()
   params = unique(data %>% filter(predictability_weight==1, grepl("_TPL", Script)) %>% select(deletion_rate, predictability_weight))
   for(i in (1:nrow(params))) {
        del = params$deletion_rate[[i]]
        pred = params$predictability_weight[[i]]
        data_ = data %>% filter(Experiment=="Experiment1", !grepl("co", Condition), predictability_weight==1, grepl("_TPL", Script), Region == "V1_0", abs(deletion_rate-del) <= 0.05, abs(predictability_weight-pred) <= 0.25)
        data_ = data_ %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
        data_$deletion_rate=del
        data_$predictability_weight=pred
        dataSmoothed = rbind(dataSmoothed, as.data.frame(data_))

   }

   plot = ggplot(dataSmoothed %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))
ggsave(plot, file="figures/model-critical-experiment1-smoothed_E0.pdf", height=1.8, width=10)




   dataSmoothed = data.frame()
   params = unique(data %>% filter(predictability_weight==1, grepl("_TPL", Script)) %>% select(deletion_rate, predictability_weight))
   for(i in (1:nrow(params))) {
        del = params$deletion_rate[[i]]
        pred = params$predictability_weight[[i]]
        data_ = data %>% filter(Experiment=="Experiment0", predictability_weight==1, grepl("_TPL", Script), Region == "V1_0", abs(deletion_rate-del) <= 0.05, abs(predictability_weight-pred) <= 0.25)
        data_ = data_ %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
        data_$deletion_rate=del
        data_$predictability_weight=pred
        dataSmoothed = rbind(dataSmoothed, as.data.frame(data_))

   }

   plot = ggplot(dataSmoothed %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))
ggsave(plot, file="figures/model-critical-experiment2-smoothed_E0.pdf", height=1.8, width=10)



data0 = data %>% filter(Experiment == "Experiment1", deletion_rate==0.5, !grepl("_co", Condition), grepl("_TPL", Script), predictability_weight==1, Region == "V1_0") %>% group_by(ID, compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(ID, compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 5.5, SurprisalReweighted <=11.5), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + facet_wrap(~ID)


data0 = data %>% filter(Experiment == "Experiment0", deletion_rate==0.5, !grepl("_TPL", Script),  predictability_weight==0.75, Region == "V1_0") %>% group_by(ID, compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(ID, compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 5.5, SurprisalReweighted <=11.5), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + facet_wrap(~ID)



data0 = data %>% filter(Experiment == "Experiment1", deletion_rate>=0.2, deletion_rate<0.8, predictability_weight==1, Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) 
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 6, SurprisalReweighted <=10.0), alpha=0.2) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 


data0 = data %>% filter(Experiment == "Experiment0", deletion_rate==0.5, predictability_weight==1, Region == "V1_0") %>% group_by(ID, compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))  %>% group_by(ID, compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
data0$ID = as.numeric(as.factor(data0$ID))
plot = ggplot(data0, aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F)  + geom_point(data = data0 %>% filter(SurprisalReweighted >= 5.5, SurprisalReweighted <=11.5), alpha=0.05) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Model Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF"))  + facet_wrap(~ID)
ggsave(plot, file="figures/model-critical-experiment2-allRuns-05-1_E0.pdf", height=7, width=7)



# Without importance reweighting (i.e., relying on the amortized approximation to the posterior): qualitative predictions are the same, despite numerical differences
plot = ggplot(data %>% filter(Experiment=="Experiment1", !grepl("_co", Condition), Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment1-full-NoLimit-NoImportanceReweighting", height=3.5, width=10)

plot = ggplot(data %>% filter(Experiment=="Experiment0", Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/model-critical-experiment2-full-NoLimit-NoImportanceReweighting", height=3.5, width=10)




