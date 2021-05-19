with open('neutral_cleaned_tokens_large.txt', 'r', encoding='utf8') as file:
    data = file.read()

list = data.split('\n')
mid = int(len(list)/2)

del list[mid:]

data = "\n".join(list)

with open('neutral_cleaned_tokens_large.txt', 'w', encoding='utf8') as file:
    file.write(data)
