import csv

list = []
with open('train.csv', newline='', encoding='utf8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list.append(row)

finalList =[]
for i in list:
    if i[1] == str(0):
        finalList += [[i[2],'NEGATIVE']]
    elif i[1] == str(1):
        finalList += [[i[2],'POSITIVE']]

with open('dataset2.csv', 'w', encoding='utf8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in finalList:
        spamwriter.writerow([i[0],i[1]])
