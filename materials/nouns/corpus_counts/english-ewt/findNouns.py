from corpusIterator_V import CorpusIterator_V

from collections import defaultdict

nounsCounts = defaultdict(int)

for sent in CorpusIterator_V("English_2.7", "train").iterator():
#    print(sent)
    for x in sent:
        if x["word"] in ["that"]:
            head = x["head"]
            if head > 0:
               verb = sent[head-1]
               head2 = verb["head"]
               if head2  > 0:
                     matrix = sent[head2-1]
                     if matrix["posUni"] == "NOUN":
                        print(matrix["lemma"], x["dep"], verb["dep"], matrix["morph"])
                        if verb["dep"] == "acl": # as opposed to acl:relcl
                              if "Number=Sing" in matrix["morph"]:
                                number = "s"
                              elif "Number=Plur" in matrix["morph"]:
                                number = "p"
                              else:
                                number = "NA"
                              nounsCounts[(matrix["lemma"], number)] += 1
#    quit()
nounsCounts = sorted(list(nounsCounts.items()), key=lambda x:x[1])
with open("output/nouns.tsv", "w") as outFile:
    for x, y in nounsCounts:
        print(f"{x[0]}\t{x[1]}\t{y}", file=outFile)
