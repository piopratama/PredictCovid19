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
# import matplotlib.pyplot as plt
import joblib
import pandas as pd
from io import BytesIO
from datetime import datetime

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

covid_dataset="./static/dataset/COVID"
non_covid_dataset="./static/dataset/NONCOVID"
hog_covid_dataset="./static/hog_dataset/COVID"
hog_non_covid_dataset="./static/hog_dataset/NONCOVID"
resize_hog_covid_dataset="./static/resize_dataset/COVID"
resize_hog_non_covid_dataset="./static/resize_dataset/NONCOVID"
k_fold_dataset='./static/'

result = Blueprint('result', __name__,
    template_folder='templates',
    static_folder='static')

@result.route('/')
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

    return render_template('result/result.html', fold=fold_folder, data_result=data_result)

@result.route('/remove', methods=['GET'])
def remove():
    p=request.args.get("p")
    if os.path.isfile("./static/model/"+p+".joblib"):
        os.remove("./static/model/"+p+".joblib")
    if os.path.isfile("./static/result/"+p+".csv"):
        os.remove("./static/result/"+p+".csv")
    if os.path.isfile("./static/result/"+p+"_result.joblib"):
        os.remove("./static/result/"+p+"_result.joblib")
    if os.path.isfile("./static/result/"+p+".png"):
        os.remove("./static/result/"+p+".png")
    if os.path.isfile("./static/result/"+p+"_detail.joblib"):
        os.remove("./static/result/"+p+"_detail.joblib")
    if os.path.isfile("./static/result/"+p+"_plot.joblib"):
        os.remove("./static/result/"+p+"_plot.joblib")
    return redirect(url_for('result.index'))

@result.route('/export_excel', methods=["GET"])
def export_excel():
    result_joblib=glob.glob("./static/result/*_result.joblib")
    data_result=[]
    table=""
    for f in result_joblib:
        d=joblib.load(f)
        data_result.append([d[0][1], d[1][1], d[2][1], d[3][1], d[4][1], d[5][1], d[6][1], d[7][1], d[8][1], d[9][1], d[10][1]])

    #create a random Pandas dataframe
    df_1 = pd.DataFrame(data_result, columns=["Experiment", "Fold", "Method", "Batch", "Learning Rate", "Epoch", "TP", "TN", "FP", "FN", "Accuracy"])

    #create an output stream
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    #taken from the original question
    df_1.to_excel(writer, startrow = 0, startcol=0, merge_cells = False, sheet_name = "Sheet_1", index=False)
    workbook = writer.book
    worksheet = writer.sheets["Sheet_1"]
    format = workbook.add_format()
    format.set_bg_color('#eeeeee')
    worksheet.set_column(0, 10, 15)

    #the writer has done its job
    writer.close()

    #go back to the beginning of the stream
    output.seek(0)

    #finally return the file
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return send_file(output, attachment_filename="result_"+dt_string+".xlsx", as_attachment=True)
    # return redirect(url_for('result.index'))