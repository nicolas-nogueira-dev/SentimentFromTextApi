
def getDataset(dataSetPath):
    # Extract the raw data
    with open(dataSetPath, "r", encoding="utf8") as file:
        data = file.read()
    # Convert the raw data into 1d list
    list = data.split("\n")
    del list[-1]
    # Convert the 1d list into 2d list
    process = []
    for i in list:
        process.append(i.split(";"))
    # Convert the 2d list into 3d list
    listSize = len(process)
    for i in range(listSize):
        process[i][0] = process[i][0].split(",")
    return process
def getTokensFromDataset(dataset):
    list = []
    for i in dataset:
        list.append(i[0])
    return list

postive = getDataset("positive_cleaned_tokens.txt")
negative = getDataset("negative_cleaned_tokens.txt")
neutral = getDataset("neutral_cleaned_tokens.txt")

neutral_tokens = getTokensFromDataset(neutral)

print(neutral[0])
print(neutral_tokens[0])
