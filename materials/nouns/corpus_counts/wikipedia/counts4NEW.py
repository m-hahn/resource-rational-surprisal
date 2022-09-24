
import gzip
                                     
import os                                                                                                                                                                                                   
nouns = []
nouns += [x for x in os.listdir("/u/scr/mhahn/nouns-that-samples/") if "_" not in x and "." not in x]
nouns+=["insinuation", "probability","conjecture", "hunch", "premonition", "estimation", "intuition", "observation", "complaint", "possibility", "anticipation", "recognition", "forecast", "projection", "demand", "instruction", "request", "proposal", "lie", "threat", "disbelief", "optimism", "requirement", "confidence", "knowledge", "doubt", "worry", "understanding", "thought"]   

strings = []
for noun in nouns:
  strings.append("the "+noun+" that")
  strings.append("the "+noun)
  strings.append("The "+noun)
  strings.append("The "+noun+" that")

count = 0
counts = [0 for _ in strings]
with gzip.open("/u/scr/mhahn/FAIR18/english-train.txt.gz", "rb") as inFile:
  for line in inFile:
     line = line.decode("utf-8")
     count += 1
     for i, string in enumerate(strings):
       if string in line:
          counts[i] += 1
     if count % 10000 == 0:
        print(count)
        u = sorted(zip(counts, strings), key=lambda x:x[0])
        print("\n".join([str(x) for x in u]))
        

u = sorted(zip(counts, strings), key=lambda x:x[0])        
with open("results/results_"+__file__+".tsv", "w") as outFile:
   print("\t".join(["Count", "Phrase", "HasThat", "Noun", "Capital"]), file=outFile)
   for count, phrase in u:
      noun = phrase.split(" ")[1]
      hasThat = (phrase.endswith("that"))
      capital = (phrase[0] != phrase[0].lower())
      print("\t".join([str(x) for x in [str(count), phrase, hasThat, noun, capital]]), file=outFile)
 
