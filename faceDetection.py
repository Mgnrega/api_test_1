import functions as f
import pickle
import cv2
import face_recognition
import json
from PIL import Image
import numpy as np
import io
import re
import base64
from io import BytesIO


classifier =pickle.loads(open('output/classifier.pickle' , 'rb').read())
lb = pickle.loads(open('output/lable_encoder.pickle' , 'rb').read())


# image = cv2.imread(f'images/Demo.jpg')

def recognise(img):

    persons = []
    image_data = re.sub('^data:image/.+;base64,', '', img)
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    # temp.show()
    image = np.array(im)
    # rgb = image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    y_pred = classifier.predict(encodings)
    y_pred = lb.inverse_transform(y_pred)
    z=0
    pred_prop = classifier.predict_proba(encodings)
    print(pred_prop)


    for ((top, right, bottom, left), name) in zip(boxes, y_pred):

            if( max(pred_prop[z]) > 0.75 ):
                # print(name)
                persons.append(name)
                print(str(max(pred_prop[z])))
                
            z+=1


    # return("hh")
    result = json.dumps(persons)
    return(result)
