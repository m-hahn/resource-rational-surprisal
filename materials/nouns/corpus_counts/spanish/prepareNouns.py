with open("output/counts_ordered.tsv", "r") as inFile:
  for line in inFile:
     if "-----" in line:
         line = line.strip().split("\t")
         print(f"topNouns.push('{line[1]}')")


