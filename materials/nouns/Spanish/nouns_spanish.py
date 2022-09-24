
topNouns = []


topNouns.append('La historia')
topNouns.append('El reporte')
topNouns.append('La ficción')
topNouns.append('La realidad')
topNouns.append('El sueño')
topNouns.append('La señal')
topNouns.append('La información')
topNouns.append('El reconocimiento')
topNouns.append('La prueba')
#topNouns.append('La impresion') // this is just a frequent typo in the corpus??!!
topNouns.append('El pensamiento')
#topNouns.append('El conocimiento')
#topNouns.append('El sentimiento')
#topNouns.append('La percepción')
#topNouns.append('La acusación')
#topNouns.append('El anuncio')
#topNouns.append('La demostración')
#topNouns.append('La confirmación')
#topNouns.append('La evidencia')
#topNouns.append('La noticia')
#topNouns.append('La posibilidad')
topNouns.append('La impresión')
topNouns.append('La sospecha')
topNouns.append('La hipótesis')
topNouns.append('La creencia')
topNouns.append('La certeza')
topNouns.append('La conclusión')
topNouns.append('El rumor')
topNouns.append('La convicción')
topNouns.append('El convencimiento')
topNouns.append('El hecho')






with open("/home/user/forgetting/corpus_counts/spanish/output/counts.tsv", "r") as inFile:
   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = counts[0]
   header = dict(list(zip(header, range(len(header)))))
   counts = counts[1:]
   counts = {x[0]: x for x in counts}


print(counts)

topNouns = list(set(topNouns))
with open("nouns_spanish.tsv", "w") as outFile:
 print("\t".join(["Noun", "Joint", "Unigram"]), file=outFile)
 for noun in topNouns:
   count = counts[noun[noun.index(" ")+1:]]
   print("\t".join([str(x) for x in [noun, int(count[1]), int(count[2])]]), file=outFile)
   print("\t".join([str(x) for x in [noun, int(count[1]), int(count[2])]]))


