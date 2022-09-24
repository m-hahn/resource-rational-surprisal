
strings = []

with open("ptb/results.tsv", "r") as inFile:
   nouns = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[1:]]
for noun in nouns:
  strings.append("the "+noun+" that")
  strings.append("the "+noun)
  strings.append("The "+noun)
  strings.append("The "+noun+" that")

count = 0
counts = [0 for _ in strings]
with open("/u/scr/mhahn/FAIR18/english-train.txt", "r") as inFile:
  for line in inFile:
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
   print("\t".join(["Count", "Phrase", "HasThat", "Noun", "Capital", "HasVerb"]), file=outFile)
   for count, phrase in u:
      noun = phrase.split(" ")[1]
      hasThat = (phrase.endswith("that"))
      hasVerb = phrase.split(" ")[-2].endswith("ing")
      capital = (phrase[0] != phrase[0].lower())
      print("\t".join([str(x) for x in [str(count), phrase, hasThat, noun, capital, hasVerb]]), file=outFile)
 
