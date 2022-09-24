import json
groups = ["trials", "system", "subject_information", "ProlificURL"]
from collections import defaultdict
trialsDataPerURL = defaultdict(list)
subjectDataPerURL = defaultdict(list)
valid, invalid = 0, 0
with open("/afs/.ir.stanford.edu/users/m/h/mhahn2/cgi-bin/experiments/serverByTrial/test.txt", "r") as inFile:
     count = 0
     for line in inFile:
          count += 1
#          print(count)
          if "103-noise-pro" not in line:
                continue
          if line.startswith("{"):
 #             print(line[:5])
              try:
                  data = json.loads(line.strip())
              except json.decoder.JSONDecodeError:
                  print("ERROR Invalid data", line)
                  invalid += 1
                  continue
 #             print(list(data))
              if "trial" in data:
                  #print(data)
                  valid += 1
                  url = data["experiment"]["ProlificURL"]
              else:
                  url = data["ProlificURL"]
              if "103-noise-pro" in url and "preview=1" not in url and "PROLIFIC_PID" in url:
                  if "trial" in data:
                      trialsDataPerURL[url].append(data["trial"])
                  else:
                      subjectDataPerURL[url].append(data)
#                  print(url)
#                  if "101-noise-pro" in url and "preview=1" not in url:
#                      dataPerGroups.append({group : data[group] for group in groups})
          elif line.startswith("=="):
              assert "REC" in line or len(set(line.strip())) == 1, line
          elif line.startswith("##"):
              assert len(set(line.strip())) == 1
          else:
             try:
                 timeStamp = float(line.strip())
             except ValueError:
                 print("ERROR", line)

def flattenedSet(l):
    s = set()
    for x in l:
        for y in x:
            s.add(y)
    return s
print(valid, invalid)
print("====================")
for x in trialsDataPerURL:
     if len(subjectDataPerURL[x]) == 0:
            print("No subject data", x, "Trials:", len(trialsDataPerURL[x]))
            subjectDataPerURL[x] = [{"ProlificURL" : x, "system" : {}, "subject_information" : {}}]
           #1, (subjectDataPerURL[x])
     assert len(subjectDataPerURL[x]) == 1
     subjectDataPerURL[x][0]["trials"] = trialsDataPerURL[x]
     subjectDataPerURL[x] = subjectDataPerURL[x][0]
dataPerGroups = sorted([y for _, y in subjectDataPerURL.items() if len(y) > 0], key=lambda x:x["ProlificURL"])
for group in groups:
  with open(group+".tsv", "w") as outFile:
    dataForGroup = [x[group] for x in dataPerGroups]
    if type(dataForGroup[0]) == type(""):
        print("\t".join(["workerid", group]), file=outFile)
        for i in range(len(dataForGroup)):
            print(("\t".join([str(i), dataForGroup[i]])).replace("\n", " "), file=outFile)
    elif type(dataForGroup[0]) == type({}):
        header = sorted(list(flattenedSet(dataForGroup)))
        print(header)
        print("\t".join(["workerid"] +header), file=outFile)
        for i in range(len( dataForGroup)):
            print(("\t".join([str(i)] + [str(dataForGroup[i].get(x, "NO_VALUE")) for x in header])).replace("\n", " "), file=outFile)
    elif type(dataForGroup[0]) == type([]):
        header = sorted(list(flattenedSet([flattenedSet(x) for x in dataForGroup])))
        print(header)
        print("\t".join(["workerid"]+header), file=outFile)
        for i in range(len(dataForGroup)):
            for j in range(len(dataForGroup[i])):
               print(("\t".join([str(i)] + [str(dataForGroup[i][j][x]) for x in header])).replace("\n", " "), file=outFile)

