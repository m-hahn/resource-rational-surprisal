from corpusIterator_V import CorpusIterator_V

from collections import defaultdict

nounsCounts = defaultdict(int)

for sent in CorpusIterator_V("Spanish_2.6", "train").iterator():
#    print(sent)
    for x in sent[:-1]:
        if x["word"] in ["de"]:
          follower = sent[x["index"]]
          if follower["head"] == x["head"] and follower["word"] == "que":
#            print("de que")
            head = x["head"]
            if head > 0:
               verb = sent[head-1]
               head2 = verb["head"]
               if head2  > 0:
                     matrix = sent[head2-1]
                     if matrix["posUni"] == "NOUN":
                        print(matrix["lemma"], x["dep"], verb["dep"])
                        if True: #verb["dep"] == "ccomp":
                              nounsCounts[matrix["lemma"]] += 1
#    quit()
nounsCounts = sorted(list(nounsCounts.items()), key=lambda x:x[1])
with open("output/nouns.tsv", "w") as outFile:
    for x, y in nounsCounts:
        print(f"{x}\t{y}", file=outFile)
