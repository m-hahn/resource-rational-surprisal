import gzip


with open("output/nouns.tsv", "r") as inFile:
    nouns = set([x.split("\t")[0] for x in inFile.read().strip().split("\n")])

from collections import defaultdict

countsTheNoun = defaultdict(int)
countsTheNounThat = defaultdict(int)

line1 = None
line2 = None
line3 = None
line4 = None
counter = 0
with gzip.open("/u/scr/mhahn/FAIR18/german-train-tagged.txt.gz", "rb") as inFile:
    for line in inFile:
        counter += 1
        if counter % 10000 == 0:
          print(counter)
        line = line.decode().strip("\n").split("\t")
        if len(line) < 3:
           assert line[0].startswith("<") and len(line) == 1, line
           continue
 #       print(line)
        line1, line2, line3, line4 = line2, line3, line4, line
        if line1 is None:
           continue
        if min([len(x) for x in [line1, line2, line3, line4]]) < 3:
           continue
       
        if line2[2] in nouns:
          if line1[2] == "die":
              countsTheNoun[line2[2]] += 1
              if line3[1] == '$,' and line4[2] in ["dass", "daß"]:
                 countsTheNounThat[line2[2]] += 1
              elif line3[2] in ["dass", "daß"]:
                 countsTheNounThat[line2[2]] += 1
        #if len(countsTheNoun) > 5:
         #  break
with open("output/counts.tsv", "w") as outFile:
    print("\t".join(["Noun", "CountWithThat", "CountWithoutThat"]), file=outFile)
    for noun in sorted(list(countsTheNoun)):
       print("\t".join([str(x) for x in [noun, countsTheNounThat.get(noun, 0), countsTheNoun[noun]]]), file=outFile)

