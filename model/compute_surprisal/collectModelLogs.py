import glob
files = sorted(glob.glob("/u/scr/mhahn/reinforce-logs-both-short/results/*"))

#print(files)

print([x for x in files if ".py" not in x])
scripts = sorted(list(set([x[x.rfind("/")+1:x.index(".py")]+".py" for x in files if ".py" in x])))

print("\n".join(scripts))


for script in scripts:
   with open(f"logsByScript/{script}.tsv", "w") as outFile:
      print(script)
      for path in [f for f in files if script in f]:
          with open(path, "r") as inFile:
            data = inFile.read().strip().split("\n")
            if "predictability_weight=1" not in data[0]:
              continue
            ID = path[path.rfind("_")+1:]
            print("\t".join([ID] + data), file=outFile)
for script in scripts:
   with open(f"logsByScript_All/{script}.tsv", "w") as outFile:
      print(script)
      for path in [f for f in files if script in f]:
          with open(path, "r") as inFile:
            data = inFile.read().strip().split("\n")
            ID = path[path.rfind("_")+1:]
            print("\t".join([ID] + data), file=outFile)

