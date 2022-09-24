import sys
import os
PATH = "/u/scr/mhahn/reinforce-logs-both-short/full-logs-retentionProbs"
files = sorted([x for x in os.listdir(PATH) if x.startswith("char")])

counts = {}
with open("/u/scr/mhahn/FAIR18/english-wiki-word-vocab.txt", "r") as inFile:
   for i in range(200000):
      line = next(inFile)
      if len(line) > 2:
        word, count = line.strip().split("\t")
      counts[word] = count

with open(PATH+f"/SUMMARY_{__file__}.tsv", "w") as outFile:
 for f in files:
   with open(PATH+"/"+f, "r") as inFile:
     args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ")])
     print(args)
     for line in inFile:
         line = line.strip().split("\t")
         word = line[0].replace("SCORES", "").strip()
         rates = line[1].strip().split(" ")
         pos = rates[-1].replace("POS=", "")
         rates = [float(q) for q in rates[:-1]]
         for i in range(len(rates)-1):
           print(word, pos, i-20, rates[i], args["myID"], args["predictability_weight"], args["deletion_rate"], counts[word], file=outFile)
          
#     quit()
