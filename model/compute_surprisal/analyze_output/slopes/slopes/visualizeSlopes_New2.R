library(ggplot2)
library(dplyr)
library(tidyr)

library(stringr)
human_slopes = read.csv("~/resource-rational-surprisal/experiments/maze/meta/output/analyze.R_slopes.tsv", sep="\t") %>% filter(!grepl("Filler", item), !grepl("Practice", item))
human_slopes[grepl("Mixed", human_slopes$item),]$compatibility = NA


model_slopes = read.csv("output/extractSlopes_Fives_WithOne_All.py.tsv", sep="\t") %>% filter(delta>=0.3, delta<=0.7) %>% group_by(item) %>% summarise(compatibility_three=mean(compatible_Three, na.rm=TRUE), embBias_three=mean(embBias_Three), compatibility_two=mean(compatible_Two, na.rm=TRUE), embBias_two=mean(embBias_Two), depth=mean(depth), intercept=mean(intercept), embBias_One=mean(embBias_One), embBias=mean(embBias), compatibility=mean(compatible))
model_slopes = model_slopes %>% mutate(item=str_replace(item, "Item", ""))


stimuli0 = read.csv("~/noisy-channel-structural-forgetting/stimuli/tsv/RTExperimentsPrevious.tsv", sep="\t")
stimuli1 = read.csv("~/noisy-channel-structural-forgetting/stimuli/tsv/Experiment1.tsv", sep="\t") %>% mutate(Sentence=NA)
stimuli2 = read.csv("~/noisy-channel-structural-forgetting/stimuli/tsv/Experiment2.tsv", sep="\t") %>% mutate(Sentence=NA) %>% filter(grepl("238_", ID)) # the other ones are redundant with stimuli0
stimuli = rbind(stimuli0, stimuli1, stimuli2)
stimuli$item = stimuli$ID
#ARCHIVE  Experiment1.tsv  Experiment2.tsv  OldModel_model.py.tsv  RTExperimentsPrevious.tsv
human_slopes = merge(human_slopes, stimuli, by=c("item"), all=TRUE)
human_slopes = human_slopes %>% filter(CriticalVerb != "") %>% mutate(Was = (CriticalVerb == "was"))
model_slopes = merge(human_slopes, model_slopes, by=c("item"), all=TRUE)

slopes = model_slopes

plot = ggplot(slopes, aes(x=embBias.x, y=compatibility.x, color=Type)) + geom_point() + geom_smooth(data=slopes %>% mutate(compatibility.x = 0), method="lm", se=F, color="black") + theme_bw()
ggsave("figures/slopes-EmbBias-Comp-ByType.pdf", height=4, width=4)
plot = ggplot(slopes %>% filter(CriticalVerb != "") %>% mutate(Was = (CriticalVerb == "was")), aes(x=embBias.x, y=compatibility.x, color=Was)) + geom_point() + geom_smooth(data=slopes %>% mutate(compatibility.x = 0), method="lm", se=F, color="black") + theme_bw()
ggsave("figures/slopes-EmbBias-Comp-ByVerb.pdf", height=4, width=4)

plot = ggplot(slopes, aes(x=embBias.y, y=compatibility.y, color=Type)) + geom_point() + geom_smooth(data=slopes %>% mutate(compatibility.x = 0), method="lm", se=F, color="black") + theme_bw()


summary(lm(compatibility_two.x ~ Was, data=slopes))
summary(lm(embBias_two.x ~ Was, data=slopes))
summary(lm(embBias_two.x ~ Type, data=slopes))



summary(lm(compatibility_three.x ~ Was, data=slopes))
summary(lm(embBias_three.x ~ Was, data=slopes))
summary(lm(embBias_three.x ~ Type, data=slopes))

summary(lm(compatibility.y ~ Was, data=slopes))
summary(lm(embBias.y ~ Was, data=slopes))
summary(lm(embBias.y ~ Type, data=slopes))

summary(lm(compatibility_three.y ~ Was, data=slopes))
summary(lm(embBias_three.y ~ Was, data=slopes))
summary(lm(embBias_three.y ~ Type, data=slopes))


cor.test(slopes$intercept.x, slopes$intercept.y)
cor.test(slopes$depth.x, slopes$depth.y)
cor.test(slopes$embBias_three.x, slopes$embBias_three.y)
cor.test(slopes$embBias_two.x, slopes$embBias_two.y)
cor.test(slopes$compatibility_three.x, slopes$compatibility_three.y)
cor.test(slopes$compatibility_two.x, slopes$compatibility_two.y)

library(ggpubr)
plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2", "Removed")), aes(x=intercept.x, y=intercept.y)) + geom_smooth(method="lm") + geom_point(aes(color=Experiment)) + theme_classic() + stat_cor(aes(color=NULL), method="spearman") + xlab("Human Log RTs") + ylab("Model Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/perItem-human-model-intercept.pdf", height=3, width=3)

plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2", "Removed")), aes(x=depth.x, y=depth.y)) + geom_smooth(method="lm") + geom_point(aes(color=Experiment)) + theme_classic() + stat_cor(aes(color=NULL), method="spearman") + xlab("Human Log RTs") + ylab("Model Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/perItem-human-model-depth.pdf", height=3, width=3)


plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2", "Removed")), aes(x=embBias_three.x, y=embBias_three.y)) + geom_smooth(method="lm") + geom_point(aes(color=Experiment)) + theme_classic() + stat_cor(aes(color=NULL), method="spearman") + xlab("Human Log RTs") + ylab("Model Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/perItem-human-model-embBias_three.pdf", height=3, width=3)

plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2", "Removed")), aes(x=embBias_two.x, y=embBias_two.y)) + geom_smooth(method="lm") + geom_point(aes(color=Experiment)) + theme_classic() + stat_cor(aes(color=NULL), method="spearman") + xlab("Human Log RTs") + ylab("Model Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/perItem-human-model-embBias_two.pdf", height=3, width=3)

plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2", "Removed")), aes(x=compatibility_three.x, y=compatibility_three.y)) + geom_smooth(method="lm") + geom_point(aes(color=Experiment)) + theme_classic() + stat_cor(aes(color=NULL), method="spearman") + xlab("Human Log RTs") + ylab("Model Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/perItem-human-model-compatibility_three.pdf", height=3, width=3)

plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2", "Removed")), aes(x=compatibility_two.x, y=compatibility_two.y)) + geom_smooth(method="lm") + geom_point(aes(color=Experiment)) + theme_classic() + stat_cor(aes(color=NULL), method="spearman") + xlab("Human Log RTs") + ylab("Model Surprisal") + theme(legend.position = "none")
ggsave(plot, file="figures/perItem-human-model-compatibility_two.pdf", height=3, width=3)


# Correlation Matrix
cor.test(slopes$intercept.y, slopes$embBias_three.y)
cor.test(slopes$intercept.x, slopes$embBias_three.x)

cor.test(slopes$intercept.x, slopes$compatibility_three.x)
cor.test(slopes$intercept.y, slopes$compatibility_three.y)

cor.test(slopes$depth.x, slopes$intercept.x)
cor.test(slopes$depth.y, slopes$intercept.y)

cor.test(slopes$embBias_three.x, slopes$compatibility_three.x)
cor.test(slopes$embBias_three.y, slopes$compatibility_three.y)


corr = function(x, y) {
	return(round(cor(x,y, use="complete"),2))
}

# Correlation Matrix
sink("output/visualizeSlopes_New2_R_correlations_human.txt")
cat(1, corr(slopes$depth.x, slopes$intercept.x), corr(slopes$intercept.x, slopes$embBias_three.x), corr(slopes$intercept.x, slopes$compatibility_three.x), "\\\\ \n", sep=" & ")
cat("-", 1, corr(slopes$depth.x, slopes$embBias_three.x), corr(slopes$depth.x, slopes$compatibility_three.x), "\\\\ \n", sep=" & ")
cat("-", "-", 1, corr(slopes$embBias_three.x, slopes$compatibility_three.x), "\\\\ \n", sep=" & ")
sink()

sink("output/visualizeSlopes_New2_R_correlations_model.txt")
cat(1, corr(slopes$depth.y, slopes$intercept.y), corr(slopes$intercept.y, slopes$embBias_three.y), corr(slopes$intercept.y, slopes$compatibility_three.y), "\\\\ \n", sep=" & ")
cat("-", 1, corr(slopes$depth.y, slopes$embBias_three.y), corr(slopes$depth.y, slopes$compatibility_three.y), "\\\\ \n", sep=" & ")
cat("-", "-", 1, corr(slopes$embBias_three.y, slopes$compatibility_three.y), "\\\\ \n", sep=" & ")
sink()



#############



cor.test(slopes$embBias.x, slopes$embBias.y)
cor.test(slopes$compatibility.x, slopes$compatibility.y)


cor.test(slopes$embBias_one, slopes$embBias_One)





plot = ggplot(slopes, aes(x=embBias_three.x, y=embBias_three.y, color=Was)) + geom_point() + theme_classic()

plot = ggplot(slopes, aes(x=depth.x, y=depth.y, color=Experiment)) + geom_point() + theme_classic()


plot = ggplot(slopes, aes(x=embBias.x, y=embBias.y, color=Type)) + geom_point() + theme_classic() + xlab("Effect in Human RTs (Log ms)") + ylab("Effect in Model Surprisal")
ggsave("figures/slopes-EmbBias-ModelHuman_byType.pdf", height=4, width=4)

plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2")), aes(x=embBias.x, y=embBias.y, color=Experiment)) + geom_point() + theme_classic() + xlab("Effect in Human RTs (Log ms)") + ylab("Effect in Model Surprisal")
ggsave("figures/slopes-EmbBias-ModelHuman_byExperiment.pdf", height=4, width=4)

plot = ggplot(slopes %>% filter(Experiment %in% c("E1", "E2")), aes(x=intercept.x, y=intercept.y, color=Experiment)) + geom_point() + theme_classic() + xlab("Effect in Human RTs (Log ms)") + ylab("Effect in Model Surprisal")
ggsave("figures/slopes-Intercept-ModelHuman_byExperiment.pdf", height=4, width=4)





plot = ggplot(slopes, aes(x=compatibility.x, y=compatibility.y, color=Was)) + geom_point() + theme_classic()
plot = ggplot(slopes, aes(x=compatibility.x, y=compatibility.y, color=Type)) + geom_point() + theme_classic()


plot = ggplot(slopes, aes(x=embBias.x, y=1, fill=Type, color=Type)) + geom_bar(stat="identity") + theme_classic()
plot = ggplot(slopes, aes(x=embBias.y, y=1, fill=Type, color=Type)) + geom_bar(stat="identity") + theme_classic()



