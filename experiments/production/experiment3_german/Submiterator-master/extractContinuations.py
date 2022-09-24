knownContinuations = set()
with open("annotated_embedded.tsv", "r") as inFile:
    for line in inFile:
        line = line.strip()
        if len(line) < 2:
            continue
        line = line.split("\t")
        assert len(line) == 2, line
        x, _ = line
        knownContinuations.add(x)
continuations = set()
with open("trials.tsv", "r") as inFile:
    header = next(inFile).strip().split("\t")
    print(header)
    header= dict(list(zip(header, range(len(header)))))
    for line in inFile:
        line = line.strip().split("\t")
        if len(line) < len(header):
            print("ERROR", line)
            continue
        if line[header["condition"]] == "SC_RC":
            if line[header["sentence1"]].split(" ")[2] == "dass":
                continue
            cont = line[header["completion"]]
            if cont not in knownContinuations:
                continuations.add(cont)
if len(continuations) > 0:
    with open("annotated_embedded.tsv", "a") as outFile:
        for x in sorted(list(continuations)):
            print(f"{x}\tNA", file=outFile)

