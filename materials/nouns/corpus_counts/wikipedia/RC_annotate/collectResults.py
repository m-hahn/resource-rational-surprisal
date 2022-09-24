import os

from collections import defaultdict                                                                                                                                                                         
queue = []                                                                                                                                                                                                  
noun, cont = None, None                                                                                                                                                                                     
useNext = True                                                                                                                                                                                              
import sys                                                                                                                                                                                                  

strings = []

#with open("../../ptb/results/results.tsv", "r") as inFile:
#   nouns = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[1:]]
nouns = [x[x.rfind("/")+1:].replace("_lab", "") for x in os.listdir("/u/scr/mhahn/nouns-that-samples/") if x.endswith("_lab")]
from collections import defaultdict
with open(f"results/{__file__}.tsv", "w") as outFile:
 print("\t".join(["Noun", "SC", "RC", "Other"]), file=outFile)
 for noun in nouns:
   overallData = defaultdict(int)
   try:
    with open("/u/scr/mhahn/nouns-that-samples/"+noun+"_lab", "r") as inFile:
     for line in inFile:
           if len(line) < 3:
               continue
           _, _, label = line.strip().split("\t")
           overallData[label] += 1
     print(overallData)
   except FileNotFoundError:
      continue 
   print("\t".join([str(x) for x in [noun, overallData["SC"], overallData["RC"], overallData["Other"]]]), file=outFile)
  
  
