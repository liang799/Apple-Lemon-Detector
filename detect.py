from PIL import Image, ImageOps
import tensorflow as tf

import cv2
import numpy as np
import os
import sys



label = ''

frame = None

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('C:\Python\Apple-Lemon-Detector\my_model.hdf5')

    
cap = cv2.VideoCapture(0)

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    prediction = import_and_predict(image, model)
    #print(prediction)

    #generate the chance of object being xxx
    probability = str(round(((np.max(prediction))*100),2))

    if np.argmax(prediction) == 0:
        result = "%   Apple"
    elif np.argmax(prediction) == 1:
        result = "%   Lemon"
    else:
        result = "%   Unknown"

    #combine these two strings
    tup1 = (probability, result)

    #convert the tuple into one string
    predict = "".join(tup1)
    
    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
