import os
import zipfile
import gzip
words = set()

with open("../previous/study5_replication/Submiterator-master/all_trials.tsv", "r") as inFile:
   for line in inFile:
      line = line.strip().split("\t")
      if len(line) > 4:
          words.add(line[5].lower().strip(".").strip(","))
with open("../experiment1/Submiterator-master/trials_byWord.tsv", "r") as inFile:
   for line in inFile:
      line = line.strip().split("\t")
      if len(line) > 4:
          words.add(line[5].lower().strip(".").strip(","))

with open("../experiment2/Submiterator-master/trials-experiment2.tsv", "r") as inFile:
   for line in inFile:
      line = line.strip().split("\t")
      if len(line) > 5:
          words.add(line[6].lower().strip(".").strip(","))
print(words)
from collections import defaultdict
frequencies = defaultdict(int)
count = 0
# http://kilgarriff.co.uk/bnc-readme.html
BASE_PATH = "/u/scr/corpora/COCA/COCA Word Lemma PoS/"
for f in os.listdir(BASE_PATH):
 with zipfile.ZipFile(BASE_PATH+f, "r") as zF:
  for f2 in zF.namelist():
   print(frequencies)
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
     word = line[0 if len(line) == 3 else 2].lower()
     if word in words:
         frequencies[word]+=1
print([x for x in words if x not in frequencies])
with open("stimuli-coca-frequencies.tsv", "w") as outFile:
 print("\t".join(["LowerCaseToken", "Frequency"]), file=outFile)
 for x, y in frequencies.items(): 
   print("\t".join([x, str(y)]), file=outFile)
