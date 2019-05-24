from __future__ import division
from __future__ import print_function

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import os
import cv2

class Detector:
	
	def __init__(self, model_def='config/yolov3.cfg', weights_path='weights/yolov3.weights', \
			class_path='data/coco.names',conf_thres=0.8, nms_thres=0.5, \
			batch_size=1, n_cpu=0, img_size=416):
		'''
			Initializes all the necessary parameters
		'''
		self.model_def = model_def
		self.weights_path = weights_path
		self.class_path = class_path
		self.conf_thres = conf_thres
		self.nms_thres = nms_thres
		self.batch_size = batch_size
		self.n_cpu = n_cpu
		self.img_size = img_size

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)
		if(self.weights_path.endswith('.weights')):
			self.model.load_darknet_weights(self.weights_path)
		else:
			self.model.load_state_dict(torch.load(self.weights_path))
		self.model.eval()

		self.classes = load_classes(self.class_path)
		self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
	
	def detect(self, image_path):
		'''
			@param image_path - Specify the path to the image location eg. image_path='./data/samples/person.jpg'
			Returns: List whose element = [class predicted, center x, center y, width, height]

			We know that the yolov3 is capable of detecting objects among the 80 classes which you can find in './data/coco.names'.
			But as we are only interested to detect the objects belonging to {'person', 'car', 'bus', 'truck'}, we filter out the
			rest of the bounding boxes.
			Alternatively, we could have utilized Transfer Learning and trained our model to detect merely the desired objects to 
			avoid the filtering process. We plan to do it later.
		'''
		imgs = []  
		img_detections = []

		# Converting the image to a torch tensor
	    	img = cv2.imread(image_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	    	img = cv2.resize(img, (self.img_size, self.img_size)) 
	    	img_ =  img[:,:,::-1].transpose((2,0,1))
	    	img_ = img_[np.newaxis,:,:,:]/255.0
	    	img_ = torch.from_numpy(img_).float()
		img_ = Variable(img_).to(self.device)

		with torch.no_grad():
	  		detections = self.model(img_)
	    		detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

                imgs.extend([image_path])
	        img_detections.extend(detections)
		for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
			if detections is not None:
				detections = rescale_boxes(detections, self.img_size, img.shape[:2])
				predictions = list()
				for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
					cls_pred = self.classes[int(cls_pred)]
					print(cls_pred)
					if cls_pred in ['car', 'bus', 'truck', 'person']:
						w = x2 - x1
						h = y2 - y1
						c_x = (x1 + x2) / 2.
						c_y = (y1 + y2) / 2.
						predictions.append([cls_pred, c_x.item(), c_y.item(), w.item(), h.item()])
				return predictions

if __name__ == '__main__':
	detector = Detector()
	objects = detector.detect('/export/livia/home/vision/vtiwari/Project/yolov3/data/samples/street.jpg')
	print('Predictions are:')
	if (objects != None and objects != []):
		for obj in objects:
			print("Class: '{}'\nCenter x-coord: {}\nCenter y-coord: {}\nWidth: {}\nHeight: {}\n---------------".format(obj[0], round(obj[1], 2), round(obj[2], 2), round(obj[3], 2), round(obj[4], 2)))
	else:
		print('Nothing DETECTED')
