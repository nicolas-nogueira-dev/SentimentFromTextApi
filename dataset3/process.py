import csv

list = []
with open('train.csv', newline='', encoding='utf8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list.append(row)

finalList =[]
for i in list:
    if i[1] == 'neg':
        finalList += [[i[0],'NEGATIVE']]
    elif i[1] == 'pos':
        finalList += [[i[0],'POSITIVE']]

with open('dataset3.csv', 'w', encoding='utf8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in finalList:
        spamwriter.writerow([i[0],i[1]])
