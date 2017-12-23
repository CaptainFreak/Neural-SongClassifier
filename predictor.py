import eyed3
import os
from os.path import isfile,join
from PIL import Image
import numpy as np
import tflearn
from classifier import createDNN

datapath="/root/deeplearning/projects/SongClassifier/Data/"


audiofile=eyed3.load(datapath+"prediction/to_predict.mp3")
if audiofile.info.mode=='Mono':
	os.system("cp "+datapath+"prediction/to_predict.mp3 /tmp/mono.mp3")
else:
	os.system("sox "+datapath+"prediction/to_predict.mp3 /tmp/mono.mp3 remix 1,2")

os.system("sox /tmp/mono.mp3 -n spectrogram -Y 200 -X 50 -m -r -o "+datapath+"prediction/spec/spec.png")  

img=Image.open(datapath+"prediction/spec/spec.png")
width,height=img.size
no_samples=int(width/128)
for x in range(no_samples):
	startpx=x*128
	imgTmp = img.crop((startpx, 1, startpx + 128, 129))
	imgTmp.save(datapath+"prediction/slices/"+str(x)+".png")


slices=[f for f in os.listdir((datapath+"prediction/slices/")) if isfile(join(datapath+"prediction/slices/",f))]



deepCNN=createDNN()
deepCNN.load('DNN.tflearn')

marwadi=0
hindi=0

for i in slices:
	img = Image.open(datapath+"prediction/slices/"+i)
	img = img.resize((128,128), resample=Image.ANTIALIAS)
	imgData = np.asarray(img, dtype=np.uint8).reshape(128,128,1)
	imgData = imgData/255.
	imgData=np.array(imgData).reshape([-1, 128, 128, 1])
	prediction=deepCNN.predict(imgData)[0]
	if prediction[0] > prediction[1]:
		hindi+=1
	else:
		marwadi+=1

			
hindi_probability=hindi*100.0/len(slices)
marwadi_probability=100.0-hindi_probability

if hindi_probability >75.0:
	print "Its a Hindi Song with probability:"+str(hindi_probability)
elif marwadi_probability >75.0:
	print "Its Marwadi Song with probability:"+str(marwadi_probability)
else:
	print "I dont know. :( "	 

