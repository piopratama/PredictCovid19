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
from werkzeug.utils import secure_filename
import cv2


covid_dataset="./static/dataset/COVID"
non_covid_dataset="./static/dataset/NONCOVID"
hog_covid_dataset="./static/hog_dataset/COVID"
hog_non_covid_dataset="./static/hog_dataset/NONCOVID"
resize_hog_covid_dataset="./static/resize_dataset/COVID"
resize_hog_non_covid_dataset="./static/resize_dataset/NONCOVID"

dashboard = Blueprint('dashboard', __name__,
    template_folder='templates',
    static_folder='static')

@dashboard.route('/')
def index():
    if request.args.get("index_page")==None:
        index_page=0
    else:
        index_page=int(request.args.get("index_page"))

    if request.args.get("index_covid")==None:
        covid=0
    elif request.args.get("index_covid")=='1':
        covid=1
    elif request.args.get("index_covid")=='2':
        covid=2
    else:
        covid=0

    if index_page<=0:
        index_page=0
    
    covid_file=glob.glob(covid_dataset+"/*.jpg")
    noncovid_file=glob.glob(non_covid_dataset+"/*.jpg")

    dataset_hog=[]
    if os.path.exists("./static/hog_dataset"):
        if os.path.exists(hog_covid_dataset) and os.path.exists(hog_non_covid_dataset):
            dataset_covid_hog=glob.glob(hog_covid_dataset+"/*.jpg")
            dataset_noncovid_hog=glob.glob(hog_non_covid_dataset+"/*.jpg")
            dataset_hog=dataset_covid_hog+dataset_noncovid_hog

    covid_file_merged = []
    noncovid_file_merged = []

    if covid==0 or covid==1:
        for f in covid_file:
            covid_file_merged.append([f, "COVID"])
    
    if covid==0 or covid==2:
        for f in noncovid_file:
            noncovid_file_merged.append([f, "NONCOVID"])

    dataset_file=covid_file_merged+noncovid_file_merged
    total_page=len(dataset_file)
    if index_page>=total_page:
        index_page=total_page-1

    total_file=len(dataset_file)
    if len(dataset_file)>=12:
        dataset_file=dataset_file[index_page*12:index_page*12+12]
    if len(dataset_hog)>=12:
        dataset_hog=dataset_hog[index_page*12:index_page*12+12]

    return render_template('dashboard/dashboard.html', dataset=dataset_file, dataset_hog=dataset_hog, index_page=index_page, total_page=total_page, total_file=total_file)

@dashboard.route('/my_hog')
def my_hog():
    if os.path.exists("./static/dataset"):
        covid_file=glob.glob(covid_dataset+"/*.jpg")
        noncovid_file=glob.glob(non_covid_dataset+"/*.jpg")

        if os.path.exists("./static/hog_dataset"):
            shutil.rmtree("./static/hog_dataset")
        
        os.mkdir("./static/hog_dataset")
        os.mkdir(hog_covid_dataset)
        os.mkdir(hog_non_covid_dataset)

        for f in covid_file:
            try:
                img = imread(f)
            #    resized_img=resize(img, (128*4,64*4))
                resized_img=img
                if len(resized_img.shape)==3:
                    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
                else:
                    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
                plt.imsave(hog_covid_dataset+"/"+os.path.basename(f), hog_image, cmap="gray")
            except:
                print("failed to process : "+f)
        
        for f in noncovid_file:
            try:
                img = imread(f)
            #    resized_img=resize(img, (128*4,64*4))
                resized_img=img
                if len(resized_img.shape)==3:
                    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
                else:
                    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
                plt.imsave(hog_non_covid_dataset+"/"+os.path.basename(f), hog_image, cmap="gray")
            except Exception as e:
                print("failed to process : "+f)

    return redirect(url_for('dashboard.index'))

@dashboard.route('/my_resize')
def my_resize():
    if os.path.exists("./static/hog_dataset"):
        covid_file=glob.glob(hog_covid_dataset+"/*.jpg")
        noncovid_file=glob.glob(hog_non_covid_dataset+"/*.jpg")
        
        if os.path.exists("./static/resize_dataset"):
            shutil.rmtree("./static/resize_dataset")
        
        os.mkdir("./static/resize_dataset")
        os.mkdir(resize_hog_covid_dataset)
        os.mkdir(resize_hog_non_covid_dataset)

        for f in covid_file:
            try:
                img = imread(f)
                resized_img=resize(img, (128,128))
                resized_img=img
                # fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
                plt.imsave(resize_hog_covid_dataset+"/"+os.path.basename(f), resized_img, cmap="gray")
            except:
                print("failed to process : "+f)
        
        for f in noncovid_file:
            try:
                img = imread(f)
                resized_img=resize(img, (128,128))
                resized_img=img
                # fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
                plt.imsave(resize_hog_non_covid_dataset+"/"+os.path.basename(f), resized_img, cmap="gray")
            except:
                print("failed to process : "+f)
    else:
        return "<h1>Please do HOG Before Resize</h1>"

    return redirect(url_for('dashboard.index'))

@dashboard.route('/my_vectorized')
def my_vectorized():
    if os.path.exists("./static/resize_dataset"):
        covid_file=glob.glob(resize_hog_covid_dataset+"/*.jpg")
        noncovid_file=glob.glob(resize_hog_non_covid_dataset+"/*.jpg")

        data=[]
        label=[]
        for f in covid_file:
            try:
                img = imread(f)
                temp=img.ravel()
                temp=temp[0::3]
                data.append(temp)
                label.append(LABEL_COVID)
                # resized_img=resize(img, (128*4,64*4))
                # resized_img=img
                # # fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
                # plt.imsave(resize_hog_covid_dataset+"/"+os.path.basename(f), resized_img, cmap="gray")
            except:
                print("failed to process : "+f)
        
        for f in noncovid_file:
            try:
                img = imread(f)
                temp=img.ravel()
                temp=temp[0::3]
                data.append(temp)
                label.append(LABEL_NON_COVID)
                # resized_img=resize(img, (128*4,64*4))
                # resized_img=img
                # # fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
                # plt.imsave(resize_hog_non_covid_dataset+"/"+os.path.basename(f), resized_img, cmap="gray")
            except:
                print("failed to process : "+f)
        
        np.savetxt("dataset.csv", data, delimiter=",")
        np.savetxt("label.csv", label, delimiter=",")
    else:
        return "<h1>Please do HOG and resize Before Vectorized</h1>"

    return redirect(url_for('dashboard.index'))

@dashboard.route('/fold_validation')
def fold_validation():
    if request.args.get("hog")==None:
        return ""

    if int(request.args.get("hog"))==1:
        if os.path.exists("./static/hog_dataset"):
            covid_file=glob.glob(hog_covid_dataset+"/*.jpg")
            noncovid_file=glob.glob(hog_non_covid_dataset+"/*.jpg")
        else:
            return "Please do HOG before doing this"
    
    if int(request.args.get("hog"))==0:
        covid_file=glob.glob(covid_dataset+"/*.jpg")
        noncovid_file=glob.glob(non_covid_dataset+"/*.jpg")

    label=[0]*len(covid_file)+[1]*len(noncovid_file)
    dataset_file=covid_file+noncovid_file

    # data=list(map(list,zip(dataset_file,label)))

    k=10
    for i in range(k):
        if os.path.exists("./static/fold"+str(i+1)):
            shutil.rmtree("./static/fold"+str(i+1))
        os.mkdir("./static/fold"+str(i+1))
        os.mkdir("./static/fold"+str(i+1)+"/train")
        os.mkdir("./static/fold"+str(i+1)+"/train/NONCOVID")
        os.mkdir("./static/fold"+str(i+1)+"/train/COVID")
        os.mkdir("./static/fold"+str(i+1)+"/test")
        os.mkdir("./static/fold"+str(i+1)+"/test/NONCOVID")
        os.mkdir("./static/fold"+str(i+1)+"/test/COVID")

    k=1
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(dataset_file, label):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = np.array(dataset_file)[train_index], np.array(dataset_file)[test_index]
        y_train, y_test = np.array(label)[train_index], np.array(label)[test_index]
        
        for x,y in zip(X_train, y_train):
            if y==0:
                copyfile(str(x), "./static/fold"+str(k)+"/train/COVID/"+os.path.basename(str(x)))
            else:
                copyfile(str(x), "./static/fold"+str(k)+"/train/NONCOVID/"+os.path.basename(str(x)))

        for x, y in zip(X_test, y_test):
            if y==0:
                copyfile(str(x), "./static/fold"+str(k)+"/test/COVID/"+os.path.basename(str(x)))
            else:
                copyfile(str(x), "./static/fold"+str(k)+"/test/NONCOVID/"+os.path.basename(str(x)))
        
        k=k+1




    return redirect(url_for('dashboard.index'))
    # return render_template('dashboard/fold_validation.html')

@dashboard.route('/upload_image', methods=["POST"])
def upload_image():
    label=request.form["label"]
    file = request.files.get('file', '')
    # npimg = np.fromstring(imagefile.read(), np.uint8)
    # img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
    # # cv2.imwrite("./static/dataset/"+label+"/"++".jpg", img)

    filename = secure_filename(file.filename)
    file.save(os.path.join(current_app.config['UPLOAD_FOLDER']+"/"+label, filename))

    return redirect(url_for('dashboard.index'))