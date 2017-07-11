import cv2

import numpy as np
import os,datetime,time , sys
print ("Package loaded") 
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from scipy.misc import  imread,imresize
import operator


print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("Done with loading packages")
print("\n")

# Load them!
cwd = os.getcwd()
loadpath = cwd + "/data/custom_data.npz"
l = np.load(loadpath)

# load an image
img1 = cv2.imread('Arnold_Schwarzenegger_0003.jpg')
img2 = cv2.imread('38.png')
img3 = cv2.imread('5.png')
img4 = cv2.imread('27.png')
img5 = cv2.imread('George_W_Bush_0006.jpg')

# See what's in here
print (l.files)

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
imgsize = l['imgsize']
use_gray = l['use_gray']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]

print("\n")
print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("Image size is %s" % (imgsize))
print ("%d classes" % (nclass))

#define variables
tf.set_random_seed(0)
n_input  = dim
n_output = nclass
if use_gray:
    weights  = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 128], stddev=0.1),name="wc1"),
        'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1),name="wc2"),
        'wd1': tf.Variable(tf.random_normal(
                [(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1),name="wd1"),
        'wd2': tf.Variable(tf.random_normal([128, n_output], stddev=0.1),name="wd2")
    }
else:
    weights  = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 128], stddev=0.1),name="wc1"),
        'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1),name="wc2"),
        'wd1': tf.Variable(tf.random_normal(
                [(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1),name="wd1"),
        'wd2': tf.Variable(tf.random_normal([128, n_output], stddev=0.1),name="wd2")
    }
biases   = {
    'bc1': tf.Variable(tf.random_normal([128], stddev=0.1),name="bc1"),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1),name="bc2"),
    'bd1': tf.Variable(tf.random_normal([128], stddev=0.1),name="bd1"),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1),name="bd2")
}

#define network
def conv_basic(_input, _w, _b, _keepratio, _use_gray):
    # INPUT
    if _use_gray:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 1])
    else:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 3])
    # CONVOLUTION LAYER 1
    _conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input_r
        , _w['wc1'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # CONVOLUTION LAYER 2
    _conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_pool_dr1
        , _w['wc2'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # VECTORIZE
    _dense1 = tf.reshape(_pool_dr2
                         , [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = {
        'out': _out
    }
    return out
print ("NETWORK READY")

#define functions
# tf Graph input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# Functions! 
_pred = conv_basic(X, weights, biases, keepratio, use_gray)['out']
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(Y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
init = tf.initialize_all_variables()

print ("FUNCTIONS READY")

# Capture Video using Webcam
cap=cv2.VideoCapture(0)

# Load CascadeClassifier to detect Faces -> frontalface.default.xml file)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




imageCounter=0
saveCount=0
piclimit = 5
faceCaptured=1


#Set the Font of the Videotext
#font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font
font = cv2.FONT_HERSHEY_SIMPLEX


print ("FUNCTIONS READY")

# Launch the graph
sess = tf.Session()
sess.run(init)
print ("FUNCTIONS READY")


#Load weights from saver 
saver = tf.train.Saver(max_to_keep=3) 
saver.restore(sess, "C:/Users/nur20/Documents/GitHub/TensorFlow-Master/save/custom_basic_cnn_idle.ckpt")
print("Model restored.")

# Capture Video using Webcam
cap=cv2.VideoCapture(0)

# Load CascadeClassifier to detect Faces -> frontalface.default.xml file)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




imageCounter=0
saveCount=0
piclimit = 5
faceCaptured=1


#Set the Font of the Videotext
#font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font
font = cv2.FONT_HERSHEY_SIMPLEX



# Launch the graph
sess = tf.Session()
sess.run(init)

print("Start restoring.")

#Load weights from saver 
saver = tf.train.Saver(max_to_keep=3) 
saver.restore(sess, "C:/Users/nur20/Documents/GitHub/TensorFlow-Master/save/custom_basic_cnn_idle.ckpt")
print("Model restored.")






while(imageCounter <200):
	#capture Frame by Frame
	ret,frame=cap.read()

	#if (imageCounter <40):
		#change the colore of a whole reagon 
	#	frame[200:450, 200:450] =img1 
	
	#if (imageCounter<80 and imageCounter >40):
	#	frame[200:450, 200:450] = img4

	#if (imageCounter <120 and imageCounter >80):
	#	frame[200:450, 200:450] = img3

	#if (imageCounter <200 and imageCounter >160):
	#	frame[200:450, 200:450] = img5

	# convert frame to frame_gray -> Frame loses its colore and gets gray
	frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#Detect face in the gray frame
	faces=face_cascade.detectMultiScale(frame_gray)
	#print("Facecoordinates: %s" %faces)



	# defines a list called crop_img 
	crop_img=[None]







	for(x,y,w,h) in faces:
		if((w>100)and(h>100)):
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

			crop_img.append(cv2.resize(frame_gray[y:y+h,x:x+w],(64,64)))
			

	
	#img_gray  = cv2.imread(imageName,0) #load grayscale
	#img_gray= rgb2gray(imread(imageName))
	img_gray_resize=imresize(frame_gray, [64, 64])/255. #resize image to [64x64]
	img_grayvec   = np.reshape(img_gray_resize, (1, -1)) #reshape matrix to vector
	#cv2.imshow("bla",img_gray_resize)
	#cv2.waitKey(1000)
	#print img_grayvec
	#print img_grayvec.shape
	imageCounter +=1
	#print(imageCounter)
	








	predictiton=sess.run(tf.nn.softmax(_pred), feed_dict={X: img_grayvec,keepratio:1.}) #make prediction
	print (predictiton)
	
	index, value = max(enumerate(predictiton[0]), key=operator.itemgetter(1)) #find highest value in output vector
	
	className=""
	if index==0:
	  className="Tyler"
	elif index ==1:
	  className="Rai"
	elif index ==2:
	  className="Bush"
	elif index ==3:
	  className="Vladimir"


	

	print ("Prediciton is class '%s' with accuracy %0.3f"%(className,value))

	if index ==0:
		cv.PutText(cv.fromarray(frame),className, (x,y-8),font, 255) #Draw the text
	elif index == 1:
		cv.PutText(cv.fromarray(frame),className, (x,y-8),font, 255) #Draw the text		
	elif index == 2:
		cv.PutText(cv.fromarray(frame),className, (x,y-8),font, 255) #Draw the text		
	elif index == 3:
		cv.PutText(cv.fromarray(frame),className, (x,y-8),font, 255) #Draw the text		
	elif index == 4:
		cv.PutText(cv.fromarray(frame),className, (x,y-8),font, 255) #Draw the text		
	elif index == 5:
		cv.PutText(cv.fromarray(frame),className, (x,y-8),font, 255) #Draw the text		



	cv2.imshow('frame',frame)

	cv2.waitKey(1)














sess.close()
print ("Session closed.")









cap.release()
cv2.destroyAllWindows()