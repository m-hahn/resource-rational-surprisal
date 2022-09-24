data = read.csv("raw_output/collect.py.tsv", sep="\t")


library(dplyr)
library(tidyr)
library(ggplot2)

for(delta in (1:19)/20) {
 for(lambda in (0:4)/4) {
    data_ = data %>% filter(deletion_rate == delta, predictability_weight == lambda)
    if(nrow(data_) < 10) {
       cat(delta, " ", lambda, "\n")
       data_ = data %>% filter(abs(deletion_rate - delta) <= 0.05, abs(predictability_weight - lambda) <= 0.25) %>% group_by(Word, POS, Distance) %>% summarise(RetentionProb = mean(RetentionProb)) %>% mutate(deletion_rate=delta, predictability_weight=lambda, Script="UNK", ID = -1)
        data = rbind(data, as.data.frame(data_))
       
    }
 }
}


data$POS = as.character(data$POS)
data$POS = ifelse(data$POS == "Det", "Determiner", data$POS)
data$POS = ifelse(data$POS == "Pron", "Pronoun", data$POS)
data$POS = ifelse(data$POS == "Prep", "Preposition", data$POS)
data$Group = ifelse(data$POS %in% c("Noun", "Verb"), "Noun/Verb", data$POS)


plot = ggplot(data %>% filter(!is.na(POS), Distance>1, Distance<10) %>% filter(Group %in% c("Noun/Verb", "Determiner", "Pronoun", "Preposition")) %>% group_by(POS, deletion_rate, predictability_weight), aes(x=Distance, y=RetentionProb, group=Group, color=Group)) +  geom_smooth(method = "glm", se = FALSE, method.args = list(family = "binomial")) + facet_grid(predictability_weight~deletion_rate) + theme_classic() + scale_x_reverse()
ggsave(plot, file="figures/grid-pos-10.pdf", height=6, width=20)


plot = ggplot(data %>% filter(predictability_weight==1, !is.na(POS), Distance>1, Distance<10) %>% filter(Group %in% c("Noun/Verb", "Determiner", "Preposition")) %>% group_by(POS, deletion_rate, predictability_weight), aes(x=Distance, y=RetentionProb, group=Group, color=Group)) +  geom_smooth(method = "glm", se = FALSE, method.args = list(family = "binomial")) + facet_grid(1~deletion_rate) + theme_classic() + scale_x_reverse()
ggsave(plot, file="figures/grid-pos-10_Lambda1.pdf", height=3, width=20)


plot = ggplot(data %>% filter(predictability_weight==1, !is.na(POS), Distance>1, Distance<10) %>% filter(Group %in% c("Noun/Verb", "Determiner", "Preposition")) %>% group_by(POS, deletion_rate, predictability_weight) %>% group_by() %>% mutate(deletion_rate=20*(1-deletion_rate)), aes(x=Distance, y=RetentionProb, group=Group, color=Group)) +  geom_smooth(method = "glm", se = FALSE, method.args = list(family = "binomial")) + facet_grid(1~deletion_rate) + theme_classic() + scale_x_reverse()
ggsave(plot, file="figures/grid-pos-10_Lambda1_Integer.pdf", height=3, width=15)

plot = ggplot(data %>% filter(predictability_weight==1, !is.na(POS), Distance>1, Distance<10) %>% filter(Group %in% c("Noun/Verb", "Determiner", "Preposition")) %>% group_by(Distance, Group, deletion_rate, predictability_weight) %>% summarise(RetentionProb=mean(RetentionProb, na.rm=TRUE)) %>% group_by() %>% mutate(deletion_rate=20*(1-deletion_rate)), aes(x=Distance, y=RetentionProb, group=Group, color=Group)) +  geom_line() + facet_grid(1~deletion_rate) + theme_classic() + scale_x_reverse()
ggsave(plot, file="figures/grid-pos-raw-10_Lambda1_Integer.pdf", height=3, width=15)

plot = ggplot(data %>% filter(predictability_weight==1, !is.na(POS), Distance>1, Distance<20) %>% filter(Group %in% c("Noun/Verb", "Determiner", "Preposition")) %>% group_by(Distance, Group, deletion_rate, predictability_weight) %>% summarise(RetentionProb=mean(RetentionProb, na.rm=TRUE)) %>% group_by() %>% mutate(deletion_rate=20*(1-deletion_rate)), aes(x=Distance, y=RetentionProb, group=Group, color=Group)) +  geom_line() + facet_grid(1~deletion_rate) + theme_classic() + scale_x_reverse()
ggsave(plot, file="figures/grid-pos-raw-20_Lambda1_Integer.pdf", height=3, width=20)










plot = ggplot(data %>% filter(!is.na(POS), Distance>1, Distance<18) %>% filter(Group %in% c("Noun/Verb", "Determiner", "Pronoun", "Preposition")) %>% group_by(POS, deletion_rate, predictability_weight), aes(x=-Distance, y=RetentionProb, group=Group, color=Group)) +  geom_smooth(method = "glm", se = FALSE, method.args = list(family = "binomial")) + facet_grid(predictability_weight~deletion_rate) + theme_classic()
ggsave(plot, file="figures/grid-pos.pdf", height=6, width=20)


plot = ggplot(data %>% filter(Distance>1) %>% group_by(POS, deletion_rate, predictability_weight, Distance) %>% summarise(RetentionProb=mean(RetentionProb)), aes(x=-Distance, y=RetentionProb, group=POS, color=POS)) + geom_line() + facet_grid(deletion_rate~predictability_weight)
ggsave(plot, file="figures/grid-pos-line.pdf", height=10, width=20)


plot = ggplot(data %>% filter(Distance>1, Distance<10) %>% group_by(POS, deletion_rate, predictability_weight, Distance) %>% summarise(RetentionProb=mean(RetentionProb)), aes(x=-Distance, y=RetentionProb, group=POS, color=POS)) + geom_line() + facet_grid(deletion_rate~predictability_weight)
ggsave(plot, file="figures/grid-pos-line-15.pdf", height=10, width=20)




plot = ggplot(data %>% filter(Distance>1, Distance<18) %>% group_by(Group, Distance) %>% summarise(RetentionProb=mean(RetentionProb)), aes(x=-Distance, y=RetentionProb, group=Group, color=Group)) + geom_line() + theme_classic() + ylim(0,1)
ggsave(plot, file="figures/grid-pos-avg.pdf", height=3, width=5)



plot = ggplot(data %>% filter(predictability_weight==1, Distance>1, Distance<18) %>% group_by(Group, Distance) %>% summarise(RetentionProb=mean(RetentionProb)), aes(x=-Distance, y=RetentionProb, group=Group, color=Group)) + geom_line() + theme_classic() + ylim(0,1)
ggsave(plot, file="figures/grid-pos-lambda1-avg.pdf", height=3, width=5)


plot = ggplot(data %>% filter(predictability_weight==1, Distance>1, Distance<10, deletion_rate>=0.2, deletion_rate<0.8, Group %in% c("Noun/Verb", "Determiner", "Pronoun", "Preposition")) %>% group_by(Group, Distance) %>% summarise(RetentionProb=mean(RetentionProb)) %>% group_by() %>% rename(POS=Group), aes(x=Distance, y=RetentionProb, group=POS, color=POS)) + geom_line() + theme_classic() + scale_x_reverse() + xlab("Distance in Past") + ylab("Retention Rate") + theme(legend.position="bottom") + guides(color = guide_legend(nrow = 2))
# + ylim(0,1)
ggsave(plot, file="figures/grid-pos-lambda1-avg.pdf", height=2.5, width=3.5)



plot = ggplot(data %>% filter(predictability_weight==1, Distance>1, Distance<10, deletion_rate>=0.2, deletion_rate<0.8, Group %in% c("Noun/Verb", "Determiner", "Pronoun", "Preposition")) %>% group_by(Group, Distance) %>% summarise(RetentionProb=mean(RetentionProb)) %>% group_by() %>% rename(POS=Group), aes(x=Distance, y=RetentionProb, group=POS, color=POS)) + geom_line() + theme_classic() + scale_x_reverse() + xlab("Distance in Past") + ylab("Retention Rate") + theme(legend.position="none")
# + ylim(0,1)
ggsave(plot, file="figures/grid-pos-lambda1-avg_NoLegend.pdf", height=2.5, width=3.5)




plot = ggplot(data %>% filter(Distance>1, Distance<18, deletion_rate==0.5, predictability_weight==0.5) %>% group_by(Group, Distance) %>% summarise(RetentionProb=mean(RetentionProb)), aes(x=-Distance, y=RetentionProb, group=Group, color=Group)) + geom_smooth(method = "glm", se = FALSE, method.args = list(family = "binomial")) + theme_classic() + ylim(0,1)
ggsave(plot, file="figures/grid-pos-05-05.pdf", height=3, width=5)





