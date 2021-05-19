def getRawDatasetComa(path):
    dataPath = path
    with  open(dataPath, 'r', encoding='utf8') as file:
        data = file.read()
    list = data.split('\n')
    dataset = []
    for i in list:
        dataset += [i.split(',')]
    return dataset

def getRawDatasetComa2(path):
    with  open(path, 'r', encoding = "ISO-8859-1") as file:
        data = file.read()
    list = data.split('\n')
    dataset = []
    for i in list:
        dataset += [i.split(',')]
    return dataset


# Dataset2
rawDataset = getRawDatasetComa('data2.txt')
del rawDataset[0]
del rawDataset[-1]

index = 0
while index != len(rawDataset):
    if rawDataset[index][1] == str(0):
        rawDataset[index][1] = "negative"
    else :
        rawDataset[index][1] = "positive"
    index += 1

# Dataset3
rawDataset3 = getRawDatasetComa2('data3.txt')
del rawDataset3[0]
del rawDataset3[-1]

index = 0
while index != len(rawDataset3):
    if rawDataset3[index][1] == str(0):
        rawDataset3[index][1] = "negative"
    else :
        rawDataset3[index][1] = "positive"
    index += 1


# Dataset1
with open("pos_tweets.txt", "r", encoding="utf8") as file:
    dataPos = file.read()
with open("neg_tweets.txt", "r", encoding="utf8") as file:
    dataNeg = file.read()

listPos = dataPos.split("\n")
listNeg = dataNeg.split("\n")

finalListPos = []
for item in listPos:
    finalListPos += [[item, "positive"]]
finalListNeg = []
for item in listNeg:
    finalListNeg += [[item, "negative"]]

index = 0
while index != len(rawDataset3):
    if rawDataset3[index][1] == str(0):
        finalListNeg.append(rawDataset3[index])
    else :
        finalListPos.append(rawDataset3[index])
    index += 1

index = 0
while index != len(rawDataset):
    if rawDataset[index][1] == str(0):
        finalListNeg.append(rawDataset[index])
    else :
        finalListPos.append(rawDataset[index])
    index += 1


textPos = []
for i in finalListPos:
    textPos.append(";".join(i))
textNeg = []
for i in finalListNeg:
    textNeg.append(";".join(i))

finalTextPos = "\n".join(textPos)
finalTextNeg = "\n".join(textNeg)

finalText = finalTextPos + finalTextNeg

with open("processed_tweets.txt", "w", encoding="utf8") as file:
    file.write(finalText)
