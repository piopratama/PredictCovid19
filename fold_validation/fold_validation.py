from flask import *
import numpy as np
import glob
import os
import shutil
from shutil import copyfile
from sklearn.model_selection import StratifiedShuffleSplit
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import joblib
import cv2
import pathlib

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.regularizers import l2
# import tensorflow as tf

covid_dataset="./static/dataset/COVID"
non_covid_dataset="./static/dataset/NONCOVID"
hog_covid_dataset="./static/hog_dataset/COVID"
hog_non_covid_dataset="./static/hog_dataset/NONCOVID"
resize_hog_covid_dataset="./static/resize_dataset/COVID"
resize_hog_non_covid_dataset="./static/resize_dataset/NONCOVID"
k_fold_dataset='./static/'

fold_validation = Blueprint('fold_validation', __name__,
    template_folder='templates',
    static_folder='static')

@fold_validation.route('/')
def index():
    static_folder=os.listdir(k_fold_dataset)
    fold_folder=[]
    for f in static_folder:
        if "fold" in f:
            fold_folder.append(f)

    result_joblib=glob.glob("./static/result/*_result.joblib")
    data_result=[]
    table=""
    for f in result_joblib:
        d=joblib.load(f)
        data_result.append(d)
        # table="<tr>"
        # table=table+"<td>"+d[0][1]+"</td>"
        # table=table+"<td>"+d[1][1]+"</td>"
        # table=table+"<td>"+d[2][1]+", "+d[3][1]+", "+d[4][1]+", "+d[5][1]+"</td>"
        # table=table+"<td>"+d[6][1]+", "+d[7][1]+", "+d[8][1]+", "+d[9][1]+", "+d[10][1]+"</td>"
        # table=table+"<td><img src='."+d[11][1]+"' class='img-center' width='150' height='150' style='cursor: pointer;'></td>"
        # table=table+"</tr>"

    return render_template('fold_validation/fold_validation.html', fold=fold_folder, data_result=data_result)

@fold_validation.route('/getFold', methods=['POST'])
def getFold():
    fold=request.form['fold']
    train0=glob.glob('./static/'+fold+"/train/NONCOVID/*.jpg")
    train1=glob.glob('./static/'+fold+"/train/COVID/*.jpg")
    test0=glob.glob('./static/'+fold+"/test/NONCOVID/*.jpg")
    test1=glob.glob('./static/'+fold+"/test/COVID/*.jpg")

    data=[]
    for f in train0:
        data.append([f, "NONCOVID", "TRAIN"])

    for f in train1:
        data.append([f, "COVID", "TRAIN"])

    for f in test0:
        data.append([f, "NONCOVID", "TEST"])

    for f in test1:
        data.append([f, "COVID", "TEST"])

    return jsonify(data)

def createTrainData(train_file, label):
    train_data=[]
    train_label=[]
    for f in train_file:
        try:
            img_array=cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img_resize_array=cv2.resize(img_array, (64, 64))
            # plt.imshow(img_resize_array, cmap="gray")
            # plt.show()
            # print(img_array)
            train_data.append(img_resize_array)
            train_label.append(label)
        except Exception as e:
            pass
    
    return train_data, train_label

def basic_model(input_shape):
    model = Sequential() 
    model.add(Conv2D(32, kernel_size = (3, 3),  
    activation = 'relu', input_shape = input_shape)) 
    model.add(Conv2D(64, (3, 3), activation = 'relu')) 
    model.add(MaxPooling2D(pool_size = (2, 2))) 
    model.add(Dropout(0.25)) 
    model.add(Flatten()) 
    model.add(Dense(128, activation = 'relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(2, activation = 'softmax'))
    return model

def vgg16(input_shape, data_train, data_test):
    model = applications.VGG16(include_top=False, weights='imagenet')
    # train_features = model.predict(data_train, len(data_train) // batch_train, verbose=1)
    # vgg16 only work with 3 channel data
    data_train=np.tile(data_train, (1, 1, 1, 3))
    data_test=np.tile(data_test, (1, 1, 1, 3))
    train_features = model.predict(data_train, verbose=1)
    # np.save('train_features.npy', train_features)
    validation_features = model.predict(data_test, verbose=1)
    # np.save('val_features.npy', validation_features)

    # train_data = np.load('train_features.npy')
    # validation_data = np.load('val_features.npy')

    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=train_features.shape[1:]))
    fc_model.add(Dense(32, activation = 'relu', kernel_regularizer=l2(0.00004)))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(2, activation='softmax'))

    return fc_model, train_features, validation_features

def MYCNN(train_path_covid, 
        train_path_non_covid, 
        test_path_covid, 
        test_path_non_covid, 
        img_width, img_height, 
        learning_rate, 
        batch_size, 
        epoch, 
        model_path, 
        experiment_name,
        fold,
        method):

    CATEGORY=["COVID", "NONCOVID"]

    train_file_covid=glob.glob(train_path_covid)
    train_file_non_covid=glob.glob(train_path_non_covid)
    test_file_covid=glob.glob(test_path_covid)
    test_file_non_covid=glob.glob(test_path_non_covid)

    train_data_covid, train_label_covid=createTrainData(train_file_covid, CATEGORY.index("COVID"))
    train_data_non_covid, train_label_non_covid=createTrainData(train_file_non_covid, CATEGORY.index("NONCOVID"))

    train_data=train_data_covid+train_data_non_covid
    train_label=train_label_covid+train_label_non_covid

    test_data_covid, test_label_covid=createTrainData(test_file_covid, CATEGORY.index("COVID"))
    test_data_non_covid, test_label_non_covid=createTrainData(test_file_non_covid, CATEGORY.index("NONCOVID"))

    test_data=test_data_covid+test_data_non_covid
    test_label=test_label_covid+test_label_non_covid

    # train_datagen = ImageDataGenerator(rescale=1./255,
    #                                shear_range=0.2,
    #                                zoom_range=0.2,
    #                                rotation_range=45,
    #                                horizontal_flip=True,
    #                                vertical_flip=True)
    # itr = train_datagen.flow_from_directory(
    #     "./static/"+fold+"/train/",
    #     target_size=(img_width, img_height),
    #     batch_size=(len(train_file_covid)+len(train_file_non_covid)),
    #     class_mode='binary',
    #     shuffle=False)

    # train_data, train_label = itr.next()

    train_data=np.array(train_data).reshape(-1, img_width, img_height, 1)
    test_data=np.array(test_data).reshape(-1, img_width, img_height, 1)

    # train_data=train_data/255.0
    # test_data=test_data/255.0

    y_train = tensorflow.keras.utils.to_categorical(train_label, 2) 
    y_test = tensorflow.keras.utils.to_categorical(test_label, 2)

    if method=="VGG16":
        print("VGG 16 Applied")
        model, train_data, test_data=vgg16(train_data.shape[1:], train_data, test_data)
    else:
        print("Basic CNN Applied")
        model=basic_model(train_data.shape[1:])

    model.compile(loss = tensorflow.keras.losses.categorical_crossentropy, 
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate), metrics = ['acc'])

    checkpoint = ModelCheckpoint(model_path+"/"+experiment_name+".joblib", verbose=1, monitor='val_loss', save_best_only=True)
    history=model.fit(
        train_data, y_train, 
        batch_size = batch_size, 
        epochs = epoch, 
        verbose = 1, 
        validation_data = (test_data, y_test),
        callbacks=[checkpoint]
    )

    score = model.evaluate(test_data, y_test, verbose = 0) 

    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])

    pred = model.predict(test_data)
    temp_predict=pred
    pred = np.argmax(pred, axis = 1) 
    label = np.argmax(y_test,axis = 1)

    # print(pred) 
    # print(label)

    # report=classification_report(label, pred)

    # print(report)

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Training', 'Validation'], loc='lower right')
    # plt.savefig('./static/result/'+experiment_name+'.png')
    # plt.close()

    confusion=confusion_matrix(label, pred)
    # report=classification_report(y_true, y_pred)
    accuracy=str(round((confusion[0][0]+confusion[1][1])*100/(confusion[0][0]+confusion[0][1]+confusion[1][0]+confusion[1][1]),2))+" %"
    result=[["Name : ", experiment_name], 
    ["Fold : ", fold], 
    ["Method : ", method], 
    ["Batch : ", str(batch_size)], 
    ["Learning Rate : ", str(learning_rate)], 
    ["Epoch : ", str(epoch)],
    ["True Positive : ", str(confusion[0][0])], 
    ["True Negative : ", str(confusion[1][1])], 
    ["False Positive : ", str(confusion[1][0])], 
    ["False Negative : ", str(confusion[0][1])], 
    ["Accuracy : ", accuracy], 
    ["image", './static/result/'+experiment_name+'.png']]

    filename_test=[]
    label_name=[]
    pred_name=[]
    path_with_name=[]
    for f in zip((test_file_covid+test_file_non_covid), label, pred):
        filename_test.append(os.path.basename(f[0]))
        label_name.append(f[1])
        pred_name.append(f[2])
        path_with_name.append(pathlib.PurePath(f[0]).parent.name+'\\'+os.path.basename(f[0]))

    detail_result=[filename_test, path_with_name, label_name, pred_name, temp_predict]

    # print(confusion)
    # print(report)

    np.savetxt("./static/result/"+experiment_name+".csv", np.array(result, dtype='str'), fmt="%s")
    joblib.dump(result, "./static/result/"+experiment_name+"_result.joblib")
    joblib.dump(detail_result, "./static/result/"+experiment_name+"_detail.joblib")
    joblib.dump([history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss']], "./static/result/"+experiment_name+"_plot.joblib")

@fold_validation.route('/train', methods=['POST'])
def train():
    fold=request.form['fold']
    method=request.form['method']
    batch=request.form['batch']
    learning_rate=request.form['learning_rate']
    name=request.form['name']
    epoch=request.form['epoch']

    if os.path.isfile("./static/result/"+name+".png"):
        os.remove("./static/result/"+name+".png")
    if os.path.isfile("./static/result/"+name+".joblib"):
        os.remove("./static/result/"+name+".joblib")
    if os.path.isfile("./static/result/"+name+".csv"):
        os.remove("./static/result/"+name+".csv")

    # CNN(name, float(learning_rate), int(batch), int(epoch), "./static/"+fold+"/train", "./static/"+fold+"/test", "./model", fold, method)
    # CNN(name, float(learning_rate), int(batch), int(epoch), "./static/"+fold+"/train", "./static/"+fold+"/test", "./model", fold, method)

    MYCNN("./static/"+fold+"/train/COVID/*.jpg", 
    "./static/"+fold+"/train/NONCOVID/*.jpg", 
    "./static/"+fold+"/test/COVID/*.jpg", 
    "./static/"+fold+"/test/NONCOVID/*.jpg", 
    64, 
    64, 
    float(learning_rate), 
    int(batch), 
    int(epoch), 
    "./static/model", 
    name,
    fold,
    method)

    return jsonify("done")