from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle as pk
import cv2

with open('./model.p','rb') as f:
    model  = pk.load(f)

img2vec = Img2Vec()

image_path ='./data/wether_dataset/test/test1.jpg'

img = Image.open(image_path)
features = img2vec.get_vec(img)
img1 = cv2.imread(image_path)
y_pred = model.predict([features])

y_pred = y_pred[0]
cv2.putText(img1,y_pred,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('frame',img1)
cv2.waitKey(5000) 
