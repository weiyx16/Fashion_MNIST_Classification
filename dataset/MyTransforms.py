import numpy as np
import random
from PIL import Image

class RandomPepperNoise (object):
	def __init__ (self,snr=0.99,p=0.5):
		self.snr=snr 
		self.p=p 
	def __call__(self,img):
		if random.uniform(0,1)<self.p:
			img_=np.array(img).copy()
			h,w=img_.shape 
			signal_pct=self.snr 
			noise_pct=(1-self.snr)
			mask=np.random.choice((0, 1, 2),size=(h,w),p=[signal_pct,noise_pct/2.,noise_pct/2.])
			img_[mask==1]=255
			img_[mask==2]=0
			return Image.fromarray(img_.astype('uint8'))
		else:
			return img