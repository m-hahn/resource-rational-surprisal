from corpusIterator_V import CorpusIterator_V

from collections import defaultdict

nounsCounts = defaultdict(int)

for sent in CorpusIterator_V("German_2.6", "train").iterator():
#    print(sent)
    for x in sent:
        if x["word"] in ["dass", "daß"]:
            head = x["head"]
            if head > 0:
               verb = sent[head-1]
               head2 = verb["head"]
               if head2  > 0:
                     matrix = sent[head2-1]
                     if matrix["posUni"] == "NOUN":
                        print(matrix["lemma"], x["dep"], verb["dep"])
                        if verb["dep"] == "ccomp":
                              nounsCounts[matrix["lemma"]] += 1
#    quit()
nounsCounts = sorted(list(nounsCounts.items()), key=lambda x:x[1])
with open("output/nouns.tsv", "w") as outFile:
    for x, y in nounsCounts:
        print(f"{x}\t{y}", file=outFile)
