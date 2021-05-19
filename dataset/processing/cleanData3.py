def getRawDatasetComa2(path):
    with  open(path, 'r', encoding = "ISO-8859-1") as file:
        data = file.read()
    list = data.split('\n')
    dataset = []
    for i in list:
        dataset += [i.split(',')]
    return dataset

rawDataset3 = getRawDatasetComa2('data3.txt')
del rawDataset3[0]
del rawDataset3[-1]

index = 0
while index != len(rawDataset3):
    if rawDataset3[index][1] == str(0):
        rawDataset3[index][0] = "negative"
    else :
        rawDataset3[index][0] = "positive"
    index += 1
