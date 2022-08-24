import json
import pickle
import io
import cv2
import numpy as np
from PIL import Image
from crypt import methods
import face_recognition
from flask import Flask , render_template , request , redirect ,flash
import faceDetection as fd
import base64
import re
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

classifier =pickle.loads(open('output/classifier.pickle' , 'rb').read())
lb = pickle.loads(open('output/lable_encoder.pickle' , 'rb').read())

@app.route("/test" , methods=['GET' , 'POST'])
def test():
    persons = []

    if request.method=='POST':

        image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
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

                if( max(pred_prop[z]) > 0.65 ):
                    # print(name)
                    persons.append(name)
                    print(str(max(pred_prop[z])))
                    
                z+=1


        # return("hh")
        result = json.dumps(persons)
        # result.headers.add('Access-Control-Allow-Origin', '*')
        return(result)


@app.route("/faceDetection" , methods=['GET' , 'POST'])
def faceDetection():
    persons = []

    if request.method=='POST':
        result = fd.recognise(request.form['imageBase64'])
        return(result)    
    
@app.route("/test2" , methods=['GET' , 'POST'])
def test2():
    return("Hello")   

@app.route("/" , methods=['GET' , 'POST'])
def base():
    return("Hello")   

if __name__ == "__main__":
        app.run(debug=True)
