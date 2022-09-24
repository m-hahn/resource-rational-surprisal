
strings = []

with open("../ptb/results/results.tsv", "r") as inFile:
   nouns = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[1:]]
import os
nouns += [x for x in os.listdir("/u/scr/mhahn/nouns-that-samples/") if "_" not in x]
nouns+=["insinuation",", ""probability","conjecture", "hunch", "premonition", "estimation", "intuition", "observation", "complaint", "possibility", "anticipation", "recognition", "forecast", "projection", "demand", "instruction", "request", "proposal", "lie", "threat", "disbelief", "optimism", "requirement", "confidence", "knowledge", "doubt", "worry", "understanding", "thought", "insight"]

nouns = set(nouns)
print(nouns)
import zipfile
import os
count = 0
from collections import defaultdict
counts_theNOUN = defaultdict(int)
counts_theNOUNthat = defaultdict(int)

buffer1, buffer2, buffer3 = None, None, None
BASE_PATH = "/u/scr/corpora/COCA/COCA Word Lemma PoS/"
for f in os.listdir(BASE_PATH):
 with zipfile.ZipFile(BASE_PATH+f, "r") as zF:
  for f2 in zF.namelist():
   print(f, f2)
   with zF.open(f2) as inFile:
    for line in inFile:
     try:
        line = line.decode("utf-8")
     except:
       print("invalid start byte")
       continue
     if line[0] in ["#", "@"]:
       continue
     line = line.split("\t")
     if len(line) not in [3,5]:
      print(line)
     buffer1 = buffer2
     buffer2 = buffer3
     buffer3 = line[0 if len(line) == 3 else 2].lower()
     #print(buffer1, buffer2, buffer3)
     if buffer1 == "the" and buffer2 in nouns:
        counts_theNOUN[buffer2] += 1
        if buffer3  == "that":
           counts_theNOUNthat[buffer2] += 1

with open("results/results_"+__file__+".tsv", "w") as outFile:
   print("\t".join(["Noun", "theNOUN", "theNOUNthat"]), file=outFile)
   for noun in sorted(list(nouns)):
      print("\t".join([str(x) for x in [noun, counts_theNOUN[noun], counts_theNOUNthat[noun]]]), file=outFile)
 
