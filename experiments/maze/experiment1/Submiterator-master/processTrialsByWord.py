import json

#{'rt': 807, 'correct': 'yes', 'word': 'patient'}

#with open("items2.txt", "r") as inFile:
#    regsPerI = [x.split("\t") for x in inFile.read().strip().split("\n")]
#regsPerI = {x[0].strip() : [x[1].strip().split(" "), x[2].strip().split(" ")] for x in regsPerI}

with open("trials_byWord.tsv", "w") as outFile:
  print("\t".join(["workerid", "condition", "item", "rt", "correct", "word", "wordInItem", "Region", "RegionFine", "sentence", "noun", "Continuation", "trial", "distractor_condition", "distractor"]), file=outFile)
  with open("trials.tsv", "r") as inFile:
    header = next(inFile).strip().split("\t")
    header = dict(list(zip(header, range(len(header)))))
    for line in inFile:
        line = line.strip().split("\t")
#        print(line)
        workerid = line[header["workerid"]]
        byWords = line[header["byWords"]]
        condition = line[header["condition"]]
        item = line[header["item"]]
        sentence = line[header["sentence"]]
        if byWords == "NA":
            continue
#        print(byWords)
        byWords = eval(byWords) #.replace("'", '"').replace("None", '"none"').replace('else"s', "else's").replace('n"t', "n't"))
        print(byWords)
 #       print(byWords, len(byWords))
        regions = [x["region"] for x in byWords]
        words = [x["word"] for x in byWords]
        if "ritic" in condition:
            regionsFine = regions[::]
#            for i in range(len(byWords) - len(regions)):
 #               regions.append("Final")
  #              regionsFine.append(f"Final_{i}")
            assert len(regions) == len(byWords), (regions, byWords)
#            print(regions)
            noun = words[1]
            continuation = "_".join(words[regions.index("REGION_3_0"):])

            print(condition, " ".join(words), continuation)
        else:
            noun, continuation = "NA", "NA"
#        if item.startswith("238_"):
#            condition_ = condition
#            condition_ = condition_.replace("incompatible", "@")
#            condition_ = condition_.replace("compatible", "incompatible")
#            condition_ = condition_.replace("@", "compatible")
#            condition = condition_

        for i in range(len(byWords)):
            print("\t".join([str(x) for x in [workerid, condition, item, byWords[i]["rt"], byWords[i]["correct"], byWords[i]["word"], i, regions[i] if "ritic" in condition else "NA", regionsFine[i] if "condition" in condition else "NA", sentence, noun, continuation, line[header["slide_number"]], "NA", byWords[i]["alt"]]]), file=outFile)
