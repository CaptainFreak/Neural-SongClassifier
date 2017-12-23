import eyed3
import os
from os.path import isfile,join
from PIL import Image
import pickle	

import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from random import shuffle


eyed3.log.setLevel("ERROR")

datapath="/root/deeplearning/projects/SongClassifier/Data/"
datapath_hindi="/root/deeplearning/projects/SongClassifier/Data/hindi/"
datapath_marwadi="/root/deeplearning/projects/SongClassifier/Data/marwadi/"

def createSpectrograms():
	os.system("mkdir "+datapath_hindi+"specs")
	os.system("mkdir "+datapath_marwadi+"specs")
	hindi_files=[f for f in os.listdir((datapath_hindi)) if isfile(join(datapath_hindi,f))]
	marwadi_files=[f for f in os.listdir((datapath_marwadi)) if isfile(join(datapath_marwadi,f))]
	
	for file in hindi_files:
		audiofile=eyed3.load(datapath_hindi+file)
		if audiofile.info.mode=='Mono':
			os.system("cp "+datapath_hindi+file+" /tmp/")
		else:
			os.system("sox "+datapath_hindi+file+" /tmp/"+file+" remix 1,2")

		os.system("sox /tmp/"+file+" -n spectrogram -Y 200 -X 50 -m -r -o "+datapath_hindi+"specs/"+file[0:-4]+".png")  

	for file in marwadi_files:
		audiofile=eyed3.load(datapath_marwadi+file)
		if audiofile.info.mode=='Mono':
			os.system("cp "+datapath_marwadi+file+" /tmp/")
		else:
			os.system("sox "+datapath_marwadi+file+" /tmp/"+file+" remix 1,2")

		os.system("sox /tmp/"+file+" -n spectrogram -Y 200 -X 50 -m -r -o "+datapath_marwadi+"specs/"+file[0:-4]+".png")        

	hindi_specs=[f for f in os.listdir((datapath_hindi+"specs")) if isfile(join(datapath_hindi+"specs",f))]
	marwadi_specs=[f for f in os.listdir((datapath_marwadi+"specs")) if isfile(join(datapath_marwadi+"specs",f))]
	


	i=0

	for spec in hindi_specs:
		img=Image.open(datapath_hindi+"specs/"+spec)
		width,height=img.size
		no_samples=int(width/128)
		for x in range(no_samples):
			startpx=x*128
			imgTmp = img.crop((startpx, 1, startpx + 128, 129))
			imgTmp.save(datapath+"slices/hindi/hindi_"+str(i)+".png")
			i+=1
	i=0 
	for spec in marwadi_specs:
		img=Image.open(datapath_marwadi+"specs/"+spec)
		width,height=img.size
		no_samples=int(width/128)
		for x in range(no_samples):
			startpx=x*128
			imgTmp = img.crop((startpx, 1, startpx + 128, 129))
			imgTmp.save(datapath+"slices/marwadi/marwadi_"+str(i)+".png")
			i+=1


def createDNN():
	print("[+] Generating Brain...")

	convnet = input_data(shape=[None, 128, 128, 1], name='input')

	convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = fully_connected(convnet, 1024, activation='elu')
	convnet = dropout(convnet, 0.5)

	convnet = fully_connected(convnet, 2, activation='softmax')
	convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

	model = tflearn.DNN(convnet,tensorboard_verbose=3)

	print("[+] Brain Created !")
	
	return model


def createDataset(mode):
	if not isfile(datapath+"dataset/train_X.p"):
		createSpectrograms()
		data=[]
		genres=['hindi','marwadi']

		for genre in genres:
			slices=[f for f in os.listdir((datapath+"slices/"+genre+"/")) if isfile(join(datapath+"slices/"+genre+"/",f))]
			shuffle(slices)
			slices=slices[:1000]
			shuffle(slices)

			for i in slices:

				img = Image.open(datapath+"slices/"+genre+"/"+i)
				img = img.resize((128,128), resample=Image.ANTIALIAS)
				imgData = np.asarray(img, dtype=np.uint8).reshape(128,128,1)
				imgData = imgData/255.
				label = [1. if genre == g else 0. for g in genres]
				data.append((imgData,label))
		
		shuffle(data)
		X,y = zip(*data)
		validationNb = int(len(X)*0.3)
		testNb = int(len(X)*0.1)
		trainNb = len(X)-(validationNb + testNb)

		train_X = np.array(X[:trainNb]).reshape([-1, 128, 128, 1])
		train_y = np.array(y[:trainNb])
		validation_X = np.array(X[trainNb:trainNb+validationNb]).reshape([-1, 128, 128, 1])
		validation_y = np.array(y[trainNb:trainNb+validationNb])
		test_X = np.array(X[-testNb:]).reshape([-1, 128, 128, 1])
		test_y = np.array(y[-testNb:])

		print "[+]Dataset Created."

		pickle.dump(train_X, open(datapath+"dataset/train_X.p", "wb" ))
		pickle.dump(train_y, open(datapath+"dataset/train_y.p", "wb" ))
		pickle.dump(validation_X, open(datapath+"dataset/validation_X.p", "wb" ))
		pickle.dump(validation_y, open(datapath+"dataset/validation_y.p", "wb" ))
		pickle.dump(test_X, open(datapath+"dataset/test_X.p", "wb" ))
		pickle.dump(test_y, open(datapath+"dataset/test_y.p", "wb" ))
		print("[+]Dataset saved!")

	if mode=="train":    
		train_X = pickle.load(open(datapath+"dataset/train_X.p", "rb" ))
		train_y = pickle.load(open(datapath+"dataset/train_y.p", "rb" ))
		validation_X = pickle.load(open(datapath+"dataset/validation_X.p", "rb" ))
		validation_y = pickle.load(open(datapath+"dataset/validation_y.p", "rb" ))
		return train_X, train_y, validation_X, validation_y	
	
	else:	    

		test_X = pickle.load(open(datapath+"dataset/test_X.p", "rb" ))
		test_y = pickle.load(open(datapath+"dataset/test_y.p", "rb" ))
		return test_X, test_y


		
if __name__ == "__main__":

	deepCNN=createDNN()

	train_X, train_y, validation_X, validation_y=createDataset("train")

	print "[+]Training DCNN--"

	deepCNN.fit(train_X, train_y, n_epoch=20, batch_size=128, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id="Training!!")

	print "[+]Train Complete--"
	print "[+]Saving weights--"

	deepCNN.save('DNN.tflearn')

	print "[+]Saved weights--"

	test_X, test_y=createDataset("test")
	deepCNN.load('DNN.tflearn')
	accuracy=deepCNN.evaluate(test_X, test_y)[0]
	print accuracy

