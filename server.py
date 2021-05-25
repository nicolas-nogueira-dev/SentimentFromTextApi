from sanic import Sanic, response, request
from TextToSentiment import *
from nltk import *

app = Sanic('TextToSentiment')
useDataset = 'dataset3/'
classifier = trainDataValidator(useDataset)
textToSentiment(classifier, 'I love chocolat')

async def home(request):
    with open('pages/home.html', 'r', encoding='utf8') as file:
        html = file.read()
    return  response.html(html)

async def process_handler(request):
    text = str(request.form.get('text'))
    predictionDict = textToSentiment(classifier, text)
    return response.json(predictionDict)

async def classifier_handler(request):
    trainClassifier(useDataset)
    classifier = trainDataValidator(useDataset)
    return response.redirect('/')

async def dataset_handler(request):
    infos = getDatasetInfos(useDataset)
    return response.json(infos)

async def choice_handler(request):
    useDataset = str(request.form.get('choice'))
    return response.redirect('/')

app.add_route(home, '/')
app.add_route(process_handler, '/process', methods=['POST'])
app.add_route(choice_handler, '/dataset', methods=['POST'])
app.add_route(classifier_handler, '/train')
app.add_route(dataset_handler, '/dataset-infos')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8000, debug=True)
