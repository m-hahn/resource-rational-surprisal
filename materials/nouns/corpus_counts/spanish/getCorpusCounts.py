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
import bz2
counter = 0
with bz2.open("/u/scr/mhahn/FAIR18/WIKIPEDIA/eswiki-20200801-pages-articles-multistream.xml.bz2", "rb") as inFile:
    for line in inFile:
        counter += 1
        if counter % 10000 == 0:
          print(counter)
          print(countsTheNounThat)
        line = line.decode()
        for noun in nouns:
           if " la "+noun in line or " el "+noun in line:
               countsTheNoun[noun] += 1
           if " la "+noun+" de que " in line or " el "+noun+" de que " in line:
               countsTheNounThat[noun] += 1
        #if len(countsTheNoun) > 5:
         #  break
with open("output/counts.tsv", "w") as outFile:
    print("\t".join(["Noun", "CountWithThat", "CountWithoutThat"]), file=outFile)
    for noun in sorted(list(countsTheNoun)):
       print("\t".join([str(x) for x in [noun, countsTheNounThat.get(noun, 0), countsTheNoun[noun]]]), file=outFile)

