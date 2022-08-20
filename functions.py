from asyncore import write
from pydoc import classname
import face_recognition
import cv2
import os
import pickle
import numpy as np

def encodings(images , classname):
    i = 0
    encode_l = []
    locations = []
    for img in images:
        print(f"Encoded {i}th / {len(images)}")
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        
        try:
            encode = face_recognition.face_encodings(img)[0]
            encode_l.append(encode)
            # loc = face_recognition.face_locations(img)[0]
            # locations.append(loc)
        except:
            print("Exception thrown x does not exist.")
            classname.pop(i)
        
        i += 1
    

    return (encode_l , locations)


def show_images( images , locations , name):

    for (img , faceloc , nam) in zip( images, locations , name):
        cv2.rectangle(img , (faceloc[3], faceloc[0]) , (faceloc[1] , faceloc[2]) , (255 , 0 , 255), 2)
        cv2.imshow(nam , img)
        cv2.waitKey(0)


def show_images_predict( encoding , images , locations , name , pr , lb):

    for (en , img , faceloc , nam) in zip( encoding , images, locations , name):
        cv2.rectangle(img , (faceloc[3], faceloc[0]) , (faceloc[1] , faceloc[2]) , (255 , 0 , 255), 2)
        y_pred= pr.predict([en])
        y_pred = lb.inverse_transform([y_pred])
        org = lb.inverse_transform([nam])
        cv2.imshow(f"{y_pred}/{org}" , img)
        # cv2.imshow(f"Image" , img)
        cv2.waitKey(0)


def get_dataset(path):
    
    images = []
    classname = []
    

    dataset = os.listdir(path)


    # for dataset

    for folder in dataset :

        list = os.listdir(path+"/"+folder)
        for pic in list:
            curimg = cv2.imread(f'{path}/{folder}/{pic}')
            images.append(curimg)
            classname.append(folder)

    return (images , classname)

def writepickle(data , file):
    f = open(file, "wb")
    f.write(pickle.dumps(data))
    f.close()

def test( X_test , y_test , classifier):
    print("Score of model : ")
    print(classifier.score(X_test , y_test))



def model(X_train , y_train ):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from catboost import CatBoostClassifier
    from sklearn.svm import SVC
    # classifier = XGBClassifier( n_estimators = 100 )
    # classifier = RandomForestClassifier( n_estimators = 300)
    classifier = CatBoostClassifier()
    # classifier = SVC(kernel='rbf' , random_state=42 , probability=  True)
    
    classifier.fit(X_train , y_train)
    return classifier


def split(encode_images , classname , testsize ):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test  = train_test_split(encode_images , classname , random_state=42 , test_size=testsize)
    return (X_train, X_test, y_train, y_test )


def create_encoder(classname):
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    classname = lb.fit_transform(classname)
    writepickle(data = lb , file='output/lable_encoder.pickle')
    return classname

def merge_encodings( path ):

    (images , classnames) = get_dataset(path)
    (encoding  , face) = encodings(images=images , classname=classnames)
    known_encodings= pickle.loads(open('output/encode_images.pickle' , 'rb').read())
    known_classnames= pickle.loads(open('output/classname.pickle' , 'rb').read())
    lb = pickle.loads(open('output/lable_encoder.pickle' , 'rb').read())

    known_classnames = lb.inverse_transform(known_classnames)
    
    print(classnames , type(classnames))
    print(known_classnames , type(known_classnames))

    known_classnames = np.ndarray.tolist(known_classnames)

    known_encodings += encoding
    for i in classnames:
        known_classnames.append(i)
    known_classnames = lb.fit_transform(known_classnames)
    classifier = model(known_encodings , known_classnames)
    writepickle(file  ='output/classifier.pickle' , data=classifier)
    writepickle( file ='output/classname.pickle' , data= known_classnames )
    writepickle( file ='output/encode_images.pickle' , data= known_encodings )
    writepickle(data = lb , file='output/lable_encoder.pickle')
