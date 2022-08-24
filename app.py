import json
import pickle
import io
import numpy as np
from PIL import Image
from crypt import methods
import face_recognition
from flask import Flask , render_template , request , redirect ,flash
import faceDetection as fd

app = Flask(__name__)

classifier =pickle.loads(open('output/classifier.pickle' , 'rb').read())
lb = pickle.loads(open('output/lable_encoder.pickle' , 'rb').read())

@app.route("/test" , methods=['GET' , 'POST'])
def test():
    persons = []

    if request.method=='POST':
        img = request.files['image']
        # print(img.file)

        dataBytesIO = io.BytesIO(img.read())
        temp = Image.open(dataBytesIO)
        # temp.show()
        image = np.array(temp)
        rgb = image

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
        return(result)


@app.route("/faceDetection" , methods=['GET' , 'POST'])
def faceDetection():
    persons = []

    if request.method=='POST':
        result = fd.recognise(request.files['image'])
        return(result)
    if request.method=='GET':
        result = fd.recognise(request.files['image'])
        return(result)    
    
@app.route("/test2" , methods=['GET' , 'POST'])
def test2():
    return("Hello")   

@app.route("/" , methods=['GET' , 'POST'])
def base():
    return("Hello")   

@app.route("/im_size", methods=['GET' , 'POST'])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})

if __name__ == "__main__":
        app.run(debug=True)
