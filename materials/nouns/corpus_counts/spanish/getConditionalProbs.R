data = read.csv("output/counts.tsv", sep="\t")

data$True_False = log(data$CountWithThat)
data$False_False = log(data$CountWithoutThat)
data$True_Minus_False = data$True_False-data$False_False


write.table(data[order(data$True_Minus_False),], file="output/counts_ordered.tsv", sep="\t", quote=FALSE)
