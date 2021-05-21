import csv

listPos = []
with open('pos_tweets.csv', newline='', encoding='utf8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
    for row in spamreader:
        listPos.append(row)

listNeg = []
with open('neg_tweets.csv', newline='', encoding='utf8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
    for row in spamreader:
        listNeg.append(row)

finalList =[]
for i in listPos:
    finalList += [[i[0],'POSITIVE']]
for i in listNeg:
    finalList += [[i[0],'NEGATIVE']]

with open('dataset4.csv', 'w', encoding='utf8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in finalList:
        spamwriter.writerow([i[0],i[1]])
