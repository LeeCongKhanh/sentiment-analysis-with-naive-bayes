from django.shortcuts import render
from django.http import HttpResponse
# from .source.da import *
# Create your views here.
from .source.da import *

def homepage(request):
    if 'sentence' in request.GET:
        sentenceRaw = request.GET['sentence']
        source_path = settings.PROJECT_ROOT+"/nbc/source/"
        nbc = NaiveBayes(source_path+"traindata.csv")
        P_all, result_predict = nbc.predict(sentenceRaw)

        return render(request, 'homepage.html', {'sentenceraw':sentenceRaw,'check':True,'p_all':P_all,'result':result_predict})
    return render(request,'homepage.html')
        
