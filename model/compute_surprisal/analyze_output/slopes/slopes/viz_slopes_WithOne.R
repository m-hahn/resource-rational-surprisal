data = read.csv("output/extractSlopes_WithOne_All.py.tsv", sep="\t")


library(ggplot2)
library(dplyr)
library(tidyr)

compat = data %>% group_by(delta, lambda) %>% summarise(t=t.test(compatible)[[1]], m=mean(compatible, na.rm=TRUE))
plot = ggplot(compat %>% filter((t)>2), aes(x=delta, y=lambda, fill=m)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse() + theme(legend.position = "none", axis.title=element_blank()) + xlim(0,1) 
ggsave("figures/effect-compatibility.pdf", width=2, height=1)

data$embBiasAcrossTwoThree = 0.5*(data$embBias_Two + data$embBias_Three)
embBias = data %>% group_by(delta, lambda) %>% summarise(t=t.test(-embBiasAcrossTwoThree)[[1]], m=mean(-embBiasAcrossTwoThree, na.rm=TRUE))
plot = ggplot(embBias %>% filter((t)>2), aes(x=delta, y=lambda, fill=m)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse() + theme(legend.position = "none", axis.title=element_blank()) + xlim(0,1) 
ggsave("figures/effect-embBias.pdf", width=2, height=1)


depth = data %>% group_by(delta, lambda) %>% summarise(t=t.test(depth)[[1]], m=mean(depth, na.rm=TRUE))
plot = ggplot(depth %>% filter((t)>2), aes(x=delta, y=lambda, fill=m)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("delta") + ylab("lambda") + guides(fill=guide_legend(title="Effect")) + scale_y_reverse() + theme(legend.position = "none", axis.title=element_blank()) + xlim(0,1)
ggsave("figures/effect-depth.pdf", width=2, height=1)



stimuli0 = read.csv("../../../../../materials/stimuli/tsv/RTExperimentsPrevious.tsv", sep="\t") %>% mutate(CriticalVerb=NULL,Sentence=NULL)
stimuli1 = read.csv("../../../../../materials/stimuli/tsv/Experiment1.tsv", sep="\t")
stimuli2 = read.csv("../../../../../materials/stimuli/tsv/Experiment2.tsv", sep="\t")

stimuli = rbind(stimuli0, stimuli1, stimuli2)

library(stringr)
stimuli = merge(stimuli %>% rename(Item=ID), data %>% mutate(Item=str_replace(item, "Item", "")), by=c("Item"))

predictions = merge(stimuli %>% filter(delta==0.5, lambda>=0.5, Experiment %in% c("E1", "E2")) %>% group_by(Experiment, lambda) %>% summarise(embBias_Three=mean(embBias_Three), embBias_Two=mean(embBias_Two), intercept=mean(intercept), embBias=mean(embBias), depth=mean(depth), compatible=mean(compatible, na.rm=TRUE), compatible.Depth=mean(compatible_Three-compatible_Two, na.rm=TRUE)), data.frame(EmbeddingBias=c(-5, 0, -5, 0, -5, 0, -5, 0)+2.5, RC=c(-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5), compatible.C = c(-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5))) %>% mutate(Surprisal = compatible*compatible.C+intercept+RC*depth+embBias*EmbeddingBias+RC*EmbeddingBias*(embBias_Three-embBias_Two) + compatible.Depth*RC*compatible.C) %>% mutate(Condition=paste(RC, compatible.C))
plot = ggplot(predictions, aes(x=EmbeddingBias, y=Surprisal, group=Condition, linetype=as.character(compatible.C), color=as.character(RC))) + geom_point() + geom_smooth(method="lm", se=F) + facet_grid(lambda~Experiment)


predictions = merge(stimuli %>% filter(delta %in% c(0.3, 0.4, 0.5, 0.6), lambda>=0.5, Experiment %in% c("E1", "E2")) %>% group_by(Experiment) %>% summarise(embBias_Three=mean(embBias_Three), embBias_Two=mean(embBias_Two), intercept=mean(intercept), embBias=mean(embBias), depth=mean(depth), compatible=mean(compatible, na.rm=TRUE), compatible.Depth=mean(compatible_Three-compatible_Two, na.rm=TRUE)), data.frame(EmbeddingBias=c(-5, 0, -5, 0, -5, 0, -5, 0)+2.5, RC=c(-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5), compatible.C = c(-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5))) %>% mutate(Surprisal = compatible*compatible.C+intercept+RC*depth+embBias*EmbeddingBias+RC*EmbeddingBias*(embBias_Three-embBias_Two) + compatible.Depth*RC*compatible.C) %>% mutate(Condition=paste(RC, compatible.C))
plot = ggplot(predictions, aes(x=EmbeddingBias, y=Surprisal, group=Condition, linetype=as.character(compatible.C), color=as.character(RC))) + geom_point() + geom_smooth(method="lm", se=F) + facet_grid(1~Experiment)





predictions = merge(stimuli %>% filter(Experiment %in% c("E1", "E2")) %>% group_by(delta, lambda) %>% summarise(embBias_Three=mean(embBias_Three), embBias_Two=mean(embBias_Two), intercept=mean(intercept), embBias=mean(embBias), depth=mean(depth), compatible=mean(compatible, na.rm=TRUE), compatible.Depth=mean(compatible_Three-compatible_Two, na.rm=TRUE)), data.frame(EmbeddingBias=c(-5, 0, -5, 0, -5, 0, -5, 0)+2.5, RC=c(-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5), compatible.C = c(-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5))) %>% mutate(Surprisal = compatible*compatible.C+intercept+RC*depth+embBias*EmbeddingBias+RC*EmbeddingBias*(embBias_Three-embBias_Two) + compatible.Depth*RC*compatible.C) %>% mutate(Condition=paste(RC, compatible.C))
plot = ggplot(predictions, aes(x=EmbeddingBias, y=Surprisal, group=Condition, linetype=as.character(compatible.C), color=as.character(RC))) + geom_point() + geom_smooth(method="lm", se=F) + facet_grid(lambda~delta)




predictions = merge(stimuli %>% filter(Experiment %in% c("E1")) %>% group_by(delta, lambda) %>% summarise(embedding=mean(embedding, na.rm=TRUE), embBias_One=mean(embBias_One, na.rm=TRUE), embBias_Three=mean(embBias_Three, na.rm=TRUE), embBias_Two=mean(embBias_Two, na.rm=TRUE), intercept=mean(intercept, na.rm=TRUE), embBias=mean(embBias, na.rm=TRUE), depth=mean(depth, na.rm=TRUE), compatible=mean(compatible, na.rm=TRUE), compatible.Depth=mean(compatible_Three-compatible_Two, na.rm=TRUE)), data.frame(EmbeddingBias=c(-5, 0, -5, 0, -5, 0, -5, 0, -5, 0)+2.5, SC = c(-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), RC=c(0, 0, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5), compatible.C = c(0, 0, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5)))
predictions = predictions %>% mutate(Surprisal = compatible*compatible.C+embedding*SC+intercept+RC*depth+embBias*EmbeddingBias-SC*(2*embBias_One-embBias)*EmbeddingBias + RC*EmbeddingBias*(embBias_Three-embBias_Two) + compatible.Depth*RC*compatible.C) %>% mutate(Condition=paste(SC, RC, compatible.C))
plot = ggplot(predictions, aes(x=EmbeddingBias, y=Surprisal, group=Condition, linetype=as.character(compatible.C), color=paste(as.character(SC), as.character(RC)))) + geom_point() + geom_smooth(method="lm", se=F) + facet_grid(lambda~delta) + theme_bw() +  theme(legend.position="none") +theme (panel.grid.major = element_blank (), panel.grid.minor = element_blank (), axis.line = element_line (colour = "black"))
ggsave(plot, file="figures/predictions-expt1-WithOne.pdf", height=3.5, width=5)



predictions = merge(stimuli %>% filter(Experiment %in% c("E2")) %>% group_by(delta, lambda) %>% summarise(embedding=mean(embedding, na.rm=TRUE), embBias_One=mean(embBias_One, na.rm=TRUE), embBias_Three=mean(embBias_Three, na.rm=TRUE), embBias_Two=mean(embBias_Two, na.rm=TRUE), intercept=mean(intercept, na.rm=TRUE), embBias=mean(embBias, na.rm=TRUE), depth=mean(depth, na.rm=TRUE), compatible=mean(compatible, na.rm=TRUE), compatible.Depth=mean(compatible_Three-compatible_Two, na.rm=TRUE)), data.frame(EmbeddingBias=c(-5, 0, -5, 0, -5, 0, -5, 0, -5, 0)+2.5, SC = c(-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), RC=c(0, 0, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5), compatible.C = c(0, 0, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5)))
predictions = predictions %>% mutate(Surprisal = compatible*compatible.C+embedding*SC+intercept+RC*depth+embBias*EmbeddingBias-SC*(2*embBias_One-embBias)*EmbeddingBias + RC*EmbeddingBias*(embBias_Three-embBias_Two) + compatible.Depth*RC*compatible.C) %>% mutate(Condition=paste(SC, RC, compatible.C))
plot = ggplot(predictions, aes(x=EmbeddingBias, y=Surprisal, group=Condition, linetype=as.character(compatible.C), color=paste(as.character(SC), as.character(RC)))) + geom_point() + geom_smooth(method="lm", se=F) + facet_grid(lambda~delta) + theme_bw() +  theme(legend.position="none") +theme (panel.grid.major = element_blank (), panel.grid.minor = element_blank (), axis.line = element_line (colour = "black"))
ggsave(plot, file="figures/predictions-expt2-WithOne.pdf", height=3.5, width=5)




predictions = merge(stimuli %>% filter(Experiment %in% c("E1", "E2")) %>% group_by(delta, lambda) %>% summarise(embedding=mean(embedding, na.rm=TRUE), embBias_One=mean(embBias_One, na.rm=TRUE), embBias_Three=mean(embBias_Three, na.rm=TRUE), embBias_Two=mean(embBias_Two, na.rm=TRUE), intercept=mean(intercept, na.rm=TRUE), embBias=mean(embBias, na.rm=TRUE), depth=mean(depth, na.rm=TRUE), compatible=mean(compatible, na.rm=TRUE), compatible.Depth=mean(compatible_Three-compatible_Two, na.rm=TRUE)), data.frame(EmbeddingBias=c(-5, 0, -5, 0, -5, 0, -5, 0, -5, 0)+2.5, SC = c(-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), RC=c(0, 0, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5), compatible.C = c(0, 0, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5)))
predictions = predictions %>% mutate(Surprisal = compatible*compatible.C+embedding*SC+intercept+RC*depth+embBias*EmbeddingBias-SC*(2*embBias_One-embBias)*EmbeddingBias + RC*EmbeddingBias*(embBias_Three-embBias_Two) + compatible.Depth*RC*compatible.C) %>% mutate(Condition=paste(SC, RC, compatible.C))
plot = ggplot(predictions, aes(x=EmbeddingBias, y=Surprisal, group=Condition, linetype=as.character(compatible.C), color=paste(as.character(SC), as.character(RC)))) + geom_point() + geom_smooth(method="lm", se=F) + facet_grid(lambda~delta) + theme_bw() +  theme(legend.position="none") +theme (panel.grid.major = element_blank (), panel.grid.minor = element_blank (), axis.line = element_line (colour = "black"))
ggsave(plot, file="figures/predictions-expt12-WithOne.pdf", height=3.5, width=5)





