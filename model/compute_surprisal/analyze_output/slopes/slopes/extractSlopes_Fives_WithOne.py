import sys

delta_, lambda_ = [(q) for q in sys.argv[1:]]

with open(f"FIXED_analyze_inter_Fives_Simple_WithOne.R_VN5_{delta_}_{lambda_}.tsv", "r") as inFile:
    data = [[z.strip('"') for z in x.split("\t")] for x in inFile.read().strip().split("\n")][1:]
for i in range(len(data)):
    data[i][0] = set(data[i][0].split(":"))
    data[i][1] = float(data[i][1])
#print(data)

#REFERENCE = "ItemMixed_1"

with open("items3.tsv", "r") as inFile:
    total_items = set(inFile.read().strip().split("\n"))

for x in data:
    if len(x[0]) == 1 and list(x[0])[0].startswith("Item") and list(x[0])[0] in total_items:
       total_items.remove(list(x[0])[0])

#print(total_items)
if len(total_items) == 1:
    REFERENCE = list(total_items)[0]
    print(delta_, lambda_, "Reference", REFERENCE)
elif len(total_items) == 0:
    assert False, total_items
else:
  if "ItemCritical_2" in total_items:
    for x in list(total_items):
        if x.startswith("ItemCritical_"):
            total_items.remove(x)
    total_items.remove('Item232_Critical_31')
    total_items.remove('Item232_Critical_29')
    total_items.remove('Item232_Critical_22')
    #print("Unknown reference after removing Critical_*", total_items)
    REFERENCE = "UNKNOWN"
  if len(total_items) > 1 and "ItemMixed_16" in total_items:
    for x in list(total_items):
        if x.startswith("ItemMixed_"):
            total_items.remove(x)
    #print("Unknown reference after removing Mixed_*", total_items)
    REFERENCE = "UNKNOWN"
if len(total_items) == 1:
    REFERENCE = list(total_items)[0]
    print(delta_, lambda_, "Reference", REFERENCE)
else:
  print(delta_, lambda_, "Unknown reference", total_items)
  REFERENCE = "UNKNOWN"

def getSlope(fixed):
    slope = {}
    REFERENCE_VALUE = None
    for x in data:
        assert x[0] != "ItemMixed_0"
        if fixed.issubset(x[0]):
            if fixed == x[0] or (len(fixed) == 0 and x[0] == set(["(Intercept)"])):
                assert REFERENCE_VALUE is None
                slope[REFERENCE] = 0
                REFERENCE_VALUE = x[1]
            if len(x[0]) == len(fixed)+1:
                if (list(x[0].difference(fixed)) + ["_"])[0].startswith("Item"):
#                   print(x[0])
                   item = (list(x[0].difference(fixed)) + ["_"])[0]
                   assert item != REFERENCE
                   slope[item] = x[1]
    for x in list(slope):
        slope[x] += REFERENCE_VALUE
    #print(sorted(list(slope.items())))
    #print(REFERENCE_VALUE)
    return slope
#    print(relevant)

intercept = getSlope(set())


embedding = getSlope(set(["SC.C"]))

depth = getSlope(set(["RC.C"]))

embBias = getSlope(set(["EmbBias.C"]))
embBiasBySC = getSlope(set(["EmbBias.C", "SC.C"]))

embBiasByRC = getSlope(set(["EmbBias.C", "RC.C"]))

comp = getSlope(set(["compatible.C"]))
compByRC = getSlope(set(["compatible.C", "RC.C"]))

embByComp = getSlope(set(["EmbBias.C", "compatible.C"]))

NA = float('nan')

with open(f"output/extractSlopes_Fives_WithOne_All.py.tsv", "a") as outFile:
  for x in sorted(list(embBias)):
    if x == "UNKNOWN":
        continue
    print("\t".join([str(q) for q in [delta_, lambda_, x] + [round(w,4) for w in [intercept.get(x, NA), embedding.get(x, float('nan')), depth.get(x, NA), embBias.get(x, NA), embBias.get(x, NA)-0.5*embBiasBySC.get(x, NA), embBias.get(x, NA)+0.5*embBiasBySC.get(x, NA)-0.5*embBiasByRC.get(x, NA), embBias.get(x, NA)+0.5*embBiasBySC.get(x, NA)+0.5*embBiasByRC.get(x, NA), comp.get(x, float('nan')), comp.get(x, NA)-0.5*compByRC.get(x, NA) if x in comp else float('nan'), comp.get(x, NA)+0.5*compByRC.get(x, NA) if x in comp else float('nan'), embByComp.get(x, NA) if x in embByComp else float('nan')]]]), file=outFile)

