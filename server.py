from sanic import Sanic, response, request
from nltk import *
from core import *
from classifierCore import *
from processCore import *

datasetInfos = {'dataset1':{'name':'dataset1',
                            'sentiments':['positive',
                                          'negative',
                                          'neutral']},
                'dataset2':{'name':'dataset2',
                            'sentiments':['positive',
                                          'negative']},
                'dataset3':{'name':'dataset3',
                            'sentiments':['positive',
                                          'negative']},
                'dataset4':{'name':'dataset4',
                            'sentiments':['positive',
                                          'negative']},
                'dataset5':{'name':'dataset5',
                            'sentiments':['positive',
                                          'negative']},
}

app = Sanic('TextToSentiment')
useDataset = 'dataset5'
classifierPath = 'classifier.pickle'

extractSettings = {'filePath':'TranslatedDigikalaDataset.csv',
                        'newFilePath':str(useDataset+'.csv'),
                        'type':'csv',
                        'encoding':'utf8',
                        'delimiter':',',
                        'quotechar':'"',
                        'delimiterSave':',',
                        'quotecharSave':'"',
                        'sentimentText':{'1':'POSITIVE',
                                         '0':'NEGATIVE',
                                        },
                        'indexExtract': {'text':0,'sentiment':1},
                        'indexSaving': {'text':0,'sentiment':1},
                        }

savingSettings = {'filePath':str(useDataset+'.csv'),
                         'type':'csv',
                         'encoding':'utf8',
                         'delimiter':',',
                         'quotechar':'"',
                         'sentimentText':{'positive':'POSITIVE',
                                          'negative':'NEGATIVE',
                                          'neutral':'NEUTRAL',
                                         },
                         'stopWord':'english',
                         'indexExtract': {'text':0,'sentiment':1},
                         }

classifierCore = ClassifierCore(useDataset,datasetInfos[useDataset]['sentiments'],classifierPath)

classifier = classifierCore.trainDataValidator()

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
    useDataset = str(request.form.get('datasetChoice'))
    return response.redirect('/')

async def debug_handler(request):
    global useDataset
    infos = {'useDataset':useDataset,'sentiments':datasetInfos[useDataset]['sentiments']}
    return response.json(infos)

async def add_dataset_handler(request):
    global extractSettings
    global savingSettings
    main = ProcessCore(useDataset,datasetInfos[useDataset]['sentiments'])
    main.preProcessDataset(extractSettings)
    main.processCleanTokens(savingSettings)
    return response.redirect('/')

app.add_route(home, '/')
app.add_route(process_handler, '/process', methods=['POST'])
app.add_route(choice_handler, '/dataset', methods=['POST'])
app.add_route(classifier_handler, '/train')
app.add_route(dataset_handler, '/dataset-infos')
app.add_route(debug_handler, '/debug')
app.add_route(add_dataset_handler, '/add-dataset')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8000, debug=True)
