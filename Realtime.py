import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
f = open('model.json', 'r')
loaded_model= f.read()
f.close()
loaded_model = model_from_json(loaded_model)
loaded_model.load_weights("weight.h5")
print("Loaded model from disk")

video=cv2.VideoCapture(0)
while True:
	check,frame=video.read()
	roi=frame[0:300,0:300]
	frame=cv2.rectangle(frame,(0,0),(300,300), (0,255,0),3)
	img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (7,7), 3)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	ret, new = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	resized=cv2.resize(new,(64,64))
	img = np.float32(resized)/255
	img = np.expand_dims(img, axis=0)
	img = np.expand_dims(img, axis=-1)
	cv2.putText(frame,str(np.argmax(loaded_model.predict(img)[0])), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), lineType=cv2.LINE_AA) 
	cv2.imshow("MainFrame",frame)
	cv2.imshow("Threshold",new)
	key=cv2.waitKey(1)
	if key==ord('q'):
		break
video.release()
cv2.destroyAllWindows()
