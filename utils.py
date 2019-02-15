from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import os
import os.path
from PIL import Image, ImageFile
import sys
import numpy as np
from numpy import linalg as LA
import cv2
import operator
from sklearn.decomposition import PCA
import scipy
import math
import time
from sys import getsizeof
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from glob import glob
import shutil

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def readTraining(dataset, rotated=True, nFiles=0, debug=False):
    if (dataset == 'oxford5k' or dataset == 'paris6k'):
        path = 'data/'+dataset+'/jpg/*.jpg'
    if (dataset == 'holidays' and rotated==True):
        path = 'dataset/holidays/jpg_rotated/Db/*.jpg'
    elif (dataset == 'holidays' and rotated==False):
        path = 'dataset/holidays/jpg_reduced/Db/*.jpg'
    elif (dataset == 'Flickr1M'):
        path = 'data/Flickr1M/im*/*/*.jpg'
    DbImages = np.sort(glob(path))  #da capire se funziona con Flickr1M

    if (dataset == 'Flickr1M'):
        DbImages = DbImages[0:int(nFiles*1000)]

    return DbImages

def readTest(dataset, full=False, debug=False):
	bBox = []
	if (dataset == 'holidays'):
		if (not full): #not rotated
			path = 'dataset/holidays/jpg_reduced/query/*.jpg'
		else:
			path = 'dataset/holidays/jpg_rotated/query/*.jpg'
	elif (dataset == 'oxford5k' or dataset == 'paris6k'):
		path = 'dataset/' + dataset + '/query'
		if (not full):
			path += '_reduced/*.jpg'
		else:
			path +='/*.jpg'

	queryImages = np.sort(queryImages)

	if ((dataset=="oxford5k" or dataset=="paris6k") and full):
		print("Creation of bBox list")
		#insert elements in bBox list
		url = 'dataset/'+dataset+'/gt_files/'
		lab_filenames = np.sort(os.listdir(url))
		for e in lab_filenames:
			if e.endswith('_query.txt'):
				q_name = e[:-len('_query.txt')]
				q_data = open("{0}/{1}".format(url, e)).readline().split(" ")
				q_filename = q_data[0][5:] if q_data[0].startswith('oxc1_') else q_data[0]
				q_final = [s.rstrip() for s in q_data]
				bBox.append(q_final[1:])
		for i,q in enumerate(queryImages,0):
			img = cv2.imread(q)
			h = img.shape[0]
			w = img.shape[1]
			bBox[i][0] = float(bBox[i][0]) / w
			bBox[i][2] = float(bBox[i][2]) / w
			bBox[i][1] = float(bBox[i][1]) / h
			bBox[i][3] = float(bBox[i][3]) / h

	return queryImages,bBox

def calculateMAC(featureVector, listData): #max-pooling and l2-norm
	rows = featureVector.shape[1] * featureVector.shape[2]
	cols = featureVector.shape[3]
	features1 = np.reshape(featureVector, (rows, cols))
	features2 = np.amax(features1, axis = 0)
	features2 /= np.linalg.norm(features2, 2)
	listData.append(features2)

	return

def calculateRMAC(features, listData, L):
	W = features.shape[1]
	H = features.shape[2]
	# print("W",W,"H",H)

	for l in range(1,L+1):
		if (l==1):
			heightRegion = widthRegion = min(W,H)
			if (W<H):
				xRegions = 1
				yRegions = 2
			else:
				xRegions = 2
				yRegions = 1
		else:
			widthRegion = heightRegion = math.ceil(2*min(W,H)/(l+1))
			if (l==2):
				xRegions = 2
				yRegions = 3
			elif (l==3):
				xRegions = 3
				yRegions = 2

		if (widthRegion*xRegions < W): #not covered the image along width
			widthRegion = math.ceil(W/xRegions)
		if (heightRegion*yRegions < H):
			heightRegion = math.ceil(H/yRegions)

		coefW = W / xRegions
		coefH = H / yRegions

		# print("L:",l," w:",widthRegion," h:",heightRegion,"xRegions",xRegions,"yRegions",yRegions)

		for x in range(0,xRegions):
			for y in range(0,yRegions):
				initialX = round(x*coefW)
				initialY = round(y*coefH)
				finalX = initialX + widthRegion
				finalY = initialY + heightRegion
				if (finalX > W):
					finalX = W
					initialX = finalX - widthRegion
				if (finalY > H):
					finalY = H
					initialY =  finalY - heightRegion

				# print(" X ",initialX,":", finalX," Y ", initialY,":", finalY)

				featureRegion = features[:,initialX:finalX,initialY:finalY,:] #(old implementation)
				calculateMAC(featureRegion, listData)
	return

def resizeImg (img, i, delta):
    if delta != 0:
        w = img.size[0]
        h = img.size[1]
        newWidth = round(w + w*delta)
        newHeight = round(h + h*delta)
        img = img.resize((newWidth,newHeight))
    return img

def extractFeatures(imgs, model, RMAC, L, resolutionLevel, bBox=[], croppedActivations = False):
	listData = []
	deltas = [0, -0.25, 0.25]
	for j in tqdm(range(0,len(imgs))):
		for i in range(0, resolutionLevel):
			img = image.load_img(imgs[j])
			img = resizeImg(img,i, deltas[i])
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = model.predict(x)
			if (croppedActivations):
				startDim1 = math.floor(bBox[j][1]*features.shape[1])
				endDim1 = math.ceil(bBox[j][3]*features.shape[1])
				startDim2 = math.floor(bBox[j][0]*features.shape[2])
				endDim2 = math.floor(bBox[j][2]*features.shape[2])
				features = np.copy(features[:,startDim1:endDim1,startDim2:endDim2,:])
				# print(features.shape,"->", features2.shape)
			calculateMAC(features, listData)
			if (RMAC):
				calculateRMAC(features, listData, L)

	return listData

def learningPCA(listData):
	fudge = 1E-18
	X = np.matrix(listData)
	mean = X.mean(axis=0)
	# subtract the mean
	X = np.subtract(X, mean)
	# calc covariance matrix
	Xcov = np.dot(X.T,X)
	d,V = np.linalg.eigh(Xcov)
	D = np.diag(1. / np.sqrt(d+fudge))
	W = np.dot(np.dot(V, D), V.T)
	return W, mean

def apply_whitening(listData, Xm, W) :
	X = np.matrix(listData)
	X = np.subtract(X, Xm)
	Xnew = np.dot(X,W)
	Xnew /= LA.norm(Xnew,axis=1).reshape(Xnew.shape[0],1)
	return Xnew


def sumPooling(listData, numberImages, largeScaleRetrieval=False):
	newListData = []
	value = 0
	regions = listData.shape[0] // numberImages
	for i, elem in enumerate(listData, 1):
		value = np.add(value,elem)
		if (i%regions==0):
			value /= LA.norm(value, 2)
			newListData.append(value)
			value = 0
	if (not largeScaleRetrieval):
		print("Sum pooling of",regions,"regions. The descriptors are",len(newListData),"of shape",newListData[0].shape)
	return newListData

def extractAndWhiteningNEW(imgs, model, RMAC, L, resolutionLevel,Xm,W, limits=1000, pca=None):
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	tmpList = []
	finalList = []
	delta = 0.25
	for j in tqdm(range(0,len(imgs))):
		for i in range(0, resolutionLevel):
			img = image.load_img(imgs[j])
			img = resizeImg(img, i, delta)
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = model.predict(x)
			calculateMAC(features, tmpList)
			if (RMAC):
				calculateRMAC(features, tmpList, L)
		if ((j+1)%limits==0):
			tmpList = apply_whitening(tmpList, Xm, W)
			tmpList = sumPooling(tmpList, limits, True)
			finalList.extend(tmpList)
			tmpList = []

	print("Features len",len(finalList))
	return finalList

def write_results(url, queryImages,i, distances, DbImages, dataset, largeScaleRetrieval=False):

	if (dataset=='oxford5k' or dataset=='paris6k'):
		if not os.path.exists(url):
			os.makedirs(url)
		file_query  = open(url+"/"+os.path.basename(queryImages[i])[:-4], "w")

		for elem in distances:
			if ((elem[0]>5062 and dataset=='oxford5k') or (elem[0]>6391 and dataset=='paris6k')):
				file_query.write("Flickr1M")
			else:
				file_query.write(os.path.basename(DbImages[elem[0]])[:-4])
			file_query.write("\n")
		file_query.close()
	elif (dataset=='holidays'):
		file_query  = open(url, "a")
		queryName = os.path.basename(queryImages[i])
		file_query.write(queryName)
		for i,elem in enumerate(distances,0):
			if (elem[0]<991):
				value = os.path.basename(DbImages[elem[0]])
				if (queryName[:4]==value[:4]):
					file_query.write(" "+str(i)+" "+str(os.path.basename(DbImages[elem[0]])))
		file_query.write("\n")
		file_query.close()

	return

def calcResults(dataset, url):
	if (dataset=='paris6k' or dataset=='oxford5k'):
		os.system("results/"+dataset+"/compute_ap_all_2 "+url)
	elif (dataset=='holidays'):
		os.system("python2 dataset/Holidays/holidays_map_single.py "+url)
	return

def retrieve(queryMAC, DbMAC, topResultsQE, url, queryImages, DbImages, dataset, largeScaleRetrieval=False):
	if (dataset=='holidays'):
		file_query  = open(url, "w")
		file_query.close()
	elif (os.path.exists(url) and (dataset=='paris6k' or dataset=='oxford5k')):
		shutil.rmtree(url)

	reRank = []

	for i,q in enumerate(queryMAC,0):
		distances = {}
		qNP = np.asarray(q)
		for j,dbElem in enumerate(DbMAC,0):
			dbNP = np.asarray(dbElem)
			distances[j] = np.linalg.norm(qNP-dbNP)
		finalDict = sorted(distances.items(), key=operator.itemgetter(1))

		reRank.extend(list(finalDict)[:topResultsQE])

		write_results(url, queryImages, i, finalDict, DbImages, dataset, largeScaleRetrieval)

	calcResults(dataset, url)

	return reRank

def retrieveRegionsNEW(queryMAC, regions, topResultsQE,url, queryImages, DbImages, dataset, largeScaleRetrieval=False):
	if (dataset=='holidays'):
		file_query  = open(url, "w")
		file_query.close()
	elif (os.path.exists(url) and (dataset=='paris6k' or dataset=='oxford5k')):
		shutil.rmtree(url)

	reRank = []

	nRegions = regions.shape[0]//len(DbImages)
	for i,q in enumerate(queryMAC,0):
		distances = {}
		bestRegions = []
		qNP = np.asarray(q)
		for j,dbElem in enumerate(regions,0):
			dbNP = np.asarray(dbElem)
			indexDb = j//nRegions
			d = np.linalg.norm(qNP-dbNP)
			if (indexDb in distances):
				if (distances[indexDb][0]>d):
					distances[indexDb] = [d,j]
			else:
				distances[indexDb] = [d,j]
		finalDict = sorted(distances.items(), key=operator.itemgetter(1))
		reRank.extend(list(finalDict)[:topResultsQE])

		write_results(url, queryImages, i, finalDict, DbImages, dataset, largeScaleRetrieval)

	calcResults(dataset, url)

	return reRank

def retrieveQE(queryMAC, DbMAC, topResultsQE, url, queryImages, DbImages, reRank, dataset, largeScaleRetrieval=False):

	url += '_avgQE'
	if (dataset=='holidays'):
		file_query  = open(url, "w")
		file_query.close()
	elif (os.path.exists(url) and (dataset=='paris6k' or dataset=='oxford5k')):
		shutil.rmtree(url)

	finalReRank = []

	for i,q in enumerate(queryMAC,0):
		distances2 = {}
		qNewNP = np.asarray(q)
		for top_results in range(0,int(topResultsQE)):
			index = top_results+(topResultsQE*i)
			dbOLD = np.asarray(DbMAC[reRank[index][0]])
			qNewNP += dbOLD
		qNewNP = np.divide(qNewNP,float(topResultsQE))
		for j,dbElem in enumerate(DbMAC,0):
			dbNP = np.asarray(dbElem)
			distances2[j] = np.linalg.norm(qNewNP-dbNP)
		finalDict = sorted(distances2.items(), key=operator.itemgetter(1))

		finalReRank.extend(list(finalDict))
		write_results(url, queryImages, i, finalDict, DbImages, dataset, largeScaleRetrieval)

	calcResults(dataset, url)
	return finalReRank

def retrieveQERegionsNEW(queryMAC, regions, topResultsQE, url, queryImages, DbImages, reRank, dataset, largeScaleRetrieval=False):

	url += '_avgQE'
	if (dataset=='holidays'):
		file_query  = open(url, "w")
		file_query.close()
	elif (os.path.exists(url) and (dataset=='paris6k' or dataset=='oxford5k')):
		shutil.rmtree(url)

	finalReRank = []

	nRegions = regions.shape[0]//len(DbImages)

	for i,q in enumerate(queryMAC,0):
		distances2 = {}
		qNewNP = np.asarray(q)
		for top_results in range(0,int(topResultsQE)):
			index = top_results+(topResultsQE*i)
			dbOLD = np.asarray(regions[reRank[index][1][1]])
			qNewNP += dbOLD
		qNewNP = np.divide(qNewNP,float(topResultsQE))
		for j,dbElem in enumerate(regions,0):
			dbNP = np.asarray(dbElem)
			indexDb = j//nRegions
			d = np.linalg.norm(qNewNP-dbNP)
			if (indexDb in distances2):
				if (distances2[indexDb]>d):
					distances2[indexDb] = d
			else:
				distances2[indexDb] = d

		finalDict = sorted(distances2.items(), key=operator.itemgetter(1))
		finalReRank.extend(list(finalDict))
		write_results(url, queryImages, i, finalDict, DbImages, dataset, largeScaleRetrieval)

	calcResults(dataset, url)
	return finalReRank
