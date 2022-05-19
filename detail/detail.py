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

covid_dataset="./static/dataset/COVID"
non_covid_dataset="./static/dataset/NONCOVID"
hog_covid_dataset="./static/hog_dataset/COVID"
hog_non_covid_dataset="./static/hog_dataset/NONCOVID"
resize_hog_covid_dataset="./static/resize_dataset/COVID"
resize_hog_non_covid_dataset="./static/resize_dataset/NONCOVID"
k_fold_dataset='./static/'

detail = Blueprint('detail', __name__,
    template_folder='templates',
    static_folder='static')

@detail.route('/', methods=['GET'])
def index():
    p=request.args.get("p")
    result_joblib=glob.glob("./static/result/"+p+"_detail.joblib")
    data_result=[]
    table=""
    filename=[]
    filepath=[]
    label=[]
    prediction=[]
    prediction_percent=[]
    for f in result_joblib:
        d=joblib.load(f)
        filename=d[0]
        filepath=d[1]
        label=d[2]
        prediction=d[3]
        prediction_percent=list(np.round(np.array(d[4])*100,2))
        # table="<tr>"
        # table=table+"<td>"+d[0][1]+"</td>"
        # table=table+"<td>"+d[1][1]+"</td>"
        # table=table+"<td>"+d[2][1]+", "+d[3][1]+", "+d[4][1]+", "+d[5][1]+"</td>"
        # table=table+"<td>"+d[6][1]+", "+d[7][1]+", "+d[8][1]+", "+d[9][1]+", "+d[10][1]+"</td>"
        # table=table+"<td><img src='."+d[11][1]+"' class='img-center' width='150' height='150' style='cursor: pointer;'></td>"
        # table=table+"</tr>"

    result_joblib=glob.glob("./static/result/"+p+"_result.joblib")
    data_result=[]
    table=""
    for f in result_joblib:
        d=joblib.load(f)
        data_result.append(d)
    
    result_plot=glob.glob("./static/result/"+p+"_plot.joblib")
    y_acc=[]
    y_val_acc=[]
    x_plot=[]
    y_loss=[]
    y_val_loss=[]
    i=1
    for f in result_plot:
        d=joblib.load(f)
        for v in d[0]:
            x_plot.append(i)
            i=i+1
        y_acc.append(list(np.round(np.array(d[0])*100,2)))
        y_loss.append(list(np.round(np.array(d[2])*100,2)))
        y_val_acc.append(list(np.round(np.array(d[1])*100,2)))
        y_val_loss.append(list(np.round(np.array(d[3])*100,2)))

    return render_template('detail/detail.html', data=zip(filename, filepath, label, prediction, prediction_percent), data_result=data_result[0], x_plot=x_plot, y_acc=y_acc, y_val_acc=y_val_acc, y_loss=y_loss, y_val_loss=y_val_loss, epoch_data=zip(x_plot, y_acc[0], y_val_acc[0], y_loss[0], y_val_loss[0]))