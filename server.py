from sanic import Sanic, response, request
from nltk import *
from core import *
from classifierCore import *
from processCore import *

app = Sanic('TextToSentiment')
useDataset = 'dataset5'
sentiments = ['positive', 'negative']
classifierPath = 'classifier.pickle'

classifierCore = ClassifierCore(useDataset,sentiments)

classifier = classifierCore.trainDataValidator()

classifierCore.textToSentiment(classifier, 'I love chocolat')

async def home(request):
    with open('pages/home.html', 'r', encoding='utf8') as file:
        html = file.read()
    return  response.html(html)

async def process_handler(request):
    text = str(request.form.get('text'))
    predictionDict = classifierCore.textToSentiment(classifier, text)
    return response.json(predictionDict)

async def classifier_handler(request):
    global classifier
    classifierCore.trainClassifier()
    classifier = classifierCore.loadClassifier(classifierPath)
    return response.redirect('/')

async def dataset_handler(request):
    infos = classifierCore.getDatasetInfos()
    return response.json(infos)

async def choice_handler(request):
    global useDataset
    global sentiments
    useDataset = str(request.form.get('datasetChoice'))
    sentiments = str(request.form.get('sentimentsChoice')).split(",")
    return response.redirect('/')

async def debug_handler(request):
    global useDataset
    global sentiments
    infos = {'useDataset':useDataset,'sentiments':sentiments}
    return response.json(infos)

app.add_route(home, '/')
app.add_route(process_handler, '/process', methods=['POST'])
app.add_route(choice_handler, '/dataset', methods=['POST'])
app.add_route(classifier_handler, '/train')
app.add_route(dataset_handler, '/dataset-infos')
app.add_route(debug_handler, '/debug')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8000, debug=True)
