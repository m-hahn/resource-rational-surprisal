with open("tex/Experiment1.tsv", "w") as outFile:
    print("\\begin{enumerate}", file=outFile)
    with open("tsv/Experiment1.tsv", "r") as inFile:
        header=next(inFile).strip().split("\t")
        header = dict(list(zip(header, range(len(header)))))
        for line in inFile:
            line = line.strip().split("\t")
            if line[header["Experiment"]] == "E1":
                print("\\item The NOUN [that the "+line[header["Noun1"]]+" [who the "+line[header["Noun2"]]+" "+line[header["Verb1"]]+"] "+line[header["Verb2_Incompatible"]]+"] "+line[header["Verb3"]].strip()+". ("+line[header["Nouns"]].lower().replace(" ", ", ")+")", file=outFile)
    print("\\end{enumerate}", file=outFile)
