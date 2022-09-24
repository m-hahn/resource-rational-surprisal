import json



with open("trials_byWord.tsv", "w") as outFile:
  print("\t".join(["workerid", "condition", "item", "rt", "correct", "word", "wordInItem", "Region", "RegionFine", "sentence", "noun", "Continuation", "trial", "distractor_condition"]), file=outFile)
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
        byWords = json.loads(byWords.replace("'", '"').replace("None", '"none"').replace('else"s', "else's").replace('n"t', "n't"))
        regions = [x["region"] for x in byWords]
        words = [x["word"] for x in byWords]
        if "ritic" in condition:
            regionsFine = regions[::]
            assert len(regions) == len(byWords), (regions, byWords)
            noun = words[1]
            continuation = "_".join(words[regions.index("REGION_3_0"):])

            print(condition, " ".join(words), continuation)
        else:
            noun, continuation = "NA", "NA"
        for i in range(len(byWords)):
            print("\t".join([str(x) for x in [workerid, condition, item, byWords[i]["rt"], byWords[i]["correct"], byWords[i]["word"], i, regions[i] if "ritic" in condition else "NA", regionsFine[i] if "condition" in condition else "NA", sentence, noun, continuation, line[header["slide_number"]], line[header["distractor_condition"]]]]), file=outFile)
