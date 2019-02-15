from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
import os
import os.path
import time
import numpy as np
from numpy import linalg as LA
from utils import *

# Starting parameters
layer = 'block5_pool'  # block5_pool, res5a_branch1 , activation_43, res5c_relu
network = 'VGG16'  # VGG16, VGG19, ResNet50, ResNet101
L = 3
topResultsQE = 5
nFiles = 100
largeScaleRetrieval = False

base_model = str_to_class(network)(weights='imagenet', include_top=False, input_shape=(None,None,3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer(str(layer)).output)


d = ["oxford5k","paris6k","holidays"]
for dataset in d:
    elif (dataset == 'oxford5k'):
        topResultsQE = 8
    elif (dataset == "paris6k"):
        topResultsQE = 6
    elif (dataset == "holidays"):
        topResultsQE = 1

    print("-------------------------------------------------")
    print('Parameters')
    print('Dataset: ' + str(dataset))
    if (dataset=="paris6k" or dataset=="holidays"):
        datasetPCA = 'oxford5k'
    elif (dataset=="oxford5k"):
        datasetPCA = "paris6k"
    print('PCA dataset: ' + str(datasetPCA))
    print('Network: ' + str(network))
    print('Layer: ' + str(layer))
    print('R-MAC descriptors with ' + str(L) + ' scales')

    resolutionLevel = 3
    print('Multi-resolution activated (3 scales: original, +25%, -25% on the largest side)')


    print("Query expansion. Top results used for QE: " + str(topResultsQE))

    if (largeScaleRetrieval):
        print("Activate large scale retrieval of",nFiles,"k files")

    print("------------------------------------------------")

    url = "results/" + dataset + "/" + network + "_L" + str(L)
    savingUrl = datasetPCA + "_"+str(network)


    url += "_multiResolution_pca" + datasetPCA


    PCAImages = readTraining(datasetPCA, False,0)
    print('PCA with '+str(len(PCAImages))+' images')
    PCAMAC = extractFeatures(PCAImages, model, True, L, resolutionLevel)
    W, Xm = learningPCA(PCAMAC)
    np.save('W'+savingUrl+'.npy',W)
    np.save('Xm'+savingUrl+'.npy',Xm)

    #after first execution comment the above snippet for the creation of the matrix W e Xm, usefull for the next PCA
    #W = np.load('W' + savingUrl + '.npy')
    #Xm = np.load('Xm' + savingUrl + '.npy')

    # ------------------ DB images: reading, descripting and whitening -----------------------

    DbImages = readTraining(dataset, True)
    print('DB contains ' + str(len(DbImages)) + ' images')

    t1 = time.clock()
    DbMAC = extractFeatures(DbImages, model, True, L, resolutionLevel)
    print("PCA-whitening")
    DbMAC = apply_whitening(DbMAC, Xm, W)
    regions = np.copy(DbMAC)
    nRegions = regions.shape[0]//len(DbImages)
    DbMAC = sumPooling(DbMAC, len(DbImages), False)
    Dbtime = time.clock() - t1
    print("RMAC and PCA-whitening of terminated in",round(Dbtime),"s")

    # ------------------- query images: reading, descripting and whitening -----------------------
    queryImages, bBox = readTest(dataset, full=True)
    print('QUERY are ' + str(len(queryImages)) + ' images')

    queryMAC = extractFeatures(queryImages, model, True, L, resolutionLevel,bBox, queryVersion)
    queryMAC = apply_whitening(queryMAC, Xm, W)
    queryMAC = sumPooling(queryMAC, 55, False)
    print("Query descriptors saved!")

    retrieval1 = time.clock()
    finalReRank = retrieveRegionsNEW(queryMAC, regions, topResultsQE,url, queryImages, DbImages, dataset)
    retrieval2 = time.clock() - retrieval1
    print("AVG query time:",round(retrieval2/len(queryImages),2),"s")

    retrieval1 = time.clock()
    finalReRank2 = retrieveQERegionsNEW(queryMAC, regions, topResultsQE, url,queryImages, DbImages, finalReRank, dataset)
    retrieval2 = time.clock() - retrieval1
    print("AVG query expansion time:",round(retrieval2/len(queryImages),2),"s")

    if (largeScaleRetrieval):
        queryMAClargeScale = np.copy(queryMAC)


    # ---------- large-scale retrieval -------------------------

    if (largeScaleRetrieval):
        print("LARGE-scale retrieval")
        url += "_"+str(nFiles)+"k"
        distractorImages = readTraining("Flickr1M", False,nFiles)
        limits = nFiles*1000//20
        print("Added",len(distractorImages),"distractors from Flickr with limits",limits)
        t10 = time.clock()
        distractorsMAC = extractAndWhiteningNEW(distractorImages, model, True, L, resolutionLevel, Xm, W, limits, None)
        t11 = time.clock() - t10
        print("Features extracted in",t11,"s")
        DbMAC.extend(distractorsMAC)
        t12 = time.clock()
        finalReRank3 = retrieve(queryMAClargeScale, DbMAC, topResultsQE,url, queryImages, DbImages, dataset, True)
        t13 = time.clock() - t12
        t13 /= len(queryImages)
        print("Avg query time:",t13,"s")
        retrieveQE(queryMAClargeScale, DbMAC, topResultsQE, url, queryImages, DbImages, finalReRank3, dataset, True)
