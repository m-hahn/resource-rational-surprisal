
topNouns = []

topNouns.append('Die Klage')
topNouns.append('Der Zweifel')
topNouns.append('Der Bericht')
topNouns.append('Die Kritik')
#topNouns.append('Der Punkt')
#topNouns.append('Die Sicherheit')
topNouns.append('Die Anordnung')
topNouns.append('Die Entscheidung')
topNouns.append('Das Zeichen')
#topNouns.append('Die Schätzung')
#topNouns.append('Die Aufforderung')
topNouns.append('Die Entdeckung')
topNouns.append('Der Beleg')
#topNouns.append('Die Idee')
#topNouns.append('Die Möglichkeit')
#topNouns.append('Der Vorwurf')
#topNouns.append('Die Erfahrung')
#topNouns.append('Die Erklärung')
#topNouns.append('Die Bestätigung')
#topNouns.append('Die Spekulation')
#topNouns.append('Die Information')
#topNouns.append('Die Ankündigung')
#topNouns.append('Der Glaube')
#topNouns.append('Die Andeutung')
#topNouns.append('Der Gedanke')
#topNouns.append('Die Aussage')
#topNouns.append('Das Gefühl')
#topNouns.append('Der Eindruck')
#topNouns.append('Der Beweis')
#topNouns.append('Der Verdacht')
#topNouns.append('Das Fazit')
#topNouns.append('Die Hoffnung')
#topNouns.append('Die Nachricht')
#topNouns.append('Die Behauptung')
#topNouns.append('Das Gerücht')
#topNouns.append('Die Mitteilung')
#topNouns.append('Die Wahrscheinlichkeit')
#topNouns.append('Der Hinweis')
topNouns.append('Die Mutmaßung')
topNouns.append('Die Erkenntnis')
topNouns.append('Die Feststellung')
topNouns.append('Die Annahme')
topNouns.append('Die Vermutung')
topNouns.append('Die Befürchtung')
topNouns.append('Die Ansicht')
topNouns.append('Die Auffassung')
topNouns.append('Die Überzeugung')
#topNouns.append('Der Schluss')
topNouns.append('Die Tatsache')




with open("/home/user/forgetting/corpus_counts/german/output/counts.tsv", "r") as inFile:
   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = counts[0]
   header = dict(list(zip(header, range(len(header)))))
   counts = counts[1:]
   counts = {x[0]: x for x in counts}


print(counts)

topNouns = list(set(topNouns))
with open("nouns_german.tsv", "w") as outFile:
 print("\t".join(["Noun", "Joint", "Unigram"]), file=outFile)
 for noun in topNouns:
   count = counts[noun[noun.index(" ")+1:]]
   print("\t".join([str(x) for x in [noun, int(count[1]), int(count[2])]]), file=outFile)
   print("\t".join([str(x) for x in [noun, int(count[1]), int(count[2])]]))


