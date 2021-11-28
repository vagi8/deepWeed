# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Monday Nov 1 21:08:10 2021

@author: Mankala Vageeshan
"""
import cv2
import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()
import pickle
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import BinaryAccuracy, Precision,Recall,AUC
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16,ResNet50,InceptionV3
import tensorflow as tf
from PIL.Image import Image as PilImage
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import pandas as pd

print("Module import successfull")

#getting the tf GPU 
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

#static arguments for training
image_mode = 'normal' # 'normal', 'grey'
# model_name = 'vgg16' #'vgg16', 'resnet50', 'inceptionv3'
no_epochs = 500

path=''
dataset_pickle_file = 'weed_train_array.pkl'
model_weights_path=''
model_json_path=''
confusion_matrix_path=''
classification_report_path=''
def define_paths(model_name):
    if model_name == 'vgg16':
        model_path=path+'VGG16'
    elif model_name == 'resnet50':
        model_path=path+'ResNet50'
    elif model_name == 'inceptionv3':
        model_path=path+'InceptionV3'

    #defining the model paths
    model_weights_path=model_path+'_model.h5'
    model_json_path=model_path+'_model.json'
    confusion_matrix_path=model_path+'_confusion_matrix.png'
    classification_report_path=model_path+'_classification_report.txt'
    return model_weights_path,model_json_path,confusion_matrix_path,classification_report_path

#loading the dataset    
if os.path.exists(path+dataset_pickle_file):
    print("Loading data from pickle file")
    with open(path+dataset_pickle_file, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded")
else:
    print("File Doesn't exits")
    sys.exit()


combined_data=np.array(data)
X = []
y = []
unique_y=[]
unique_label=[]
for features,label,species_name in combined_data:
    if image_mode=='grey':
      features = cv2.cvtColor(features, cv2.COLOR_RGB2GRAY)
    X.append(features)
    y.append(label)
    if label not in unique_y:
      unique_label.append(species_name)
      unique_y.append(label)

def image_reshape(image,image_mode):
    # reshape X data to make valid 4D data with numpy
    if image_mode=='grey':
        img = np.array(image).reshape(-1, 64, 64)
    else:
        img = np.array(image).reshape(-1, 64, 64,3)
    return img

X = image_reshape(X,image_mode)

def display_images(
    images: [PilImage], labels,
    columns=5, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):

    if not images:
        print("No images to display.")
        return 

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)
        plt.title(labels[i])

print("Splitting train & test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"X Train : {X_train.shape}")
print(f"Y Train : {y_train.shape}")
print(f"X Test : {X_test.shape}")
print(f"Y Test : {y_test.shape}")
print(f"X Validation : {X_val.shape}")
print(f"Y Validation : {len(y_val)}")

def run_model(model_name):
    print("model_name - ", model_name)
    model_weights_path,model_json_path,confusion_matrix_path,classification_report_path=define_paths(model_name)
    print("model_weights_path - ", model_weights_path)
    print("model_json_path - ", model_json_path)
    print("confusion_matrix_path - ", confusion_matrix_path)
    print("classification_report_path - ", classification_report_path)
    if model_name == 'vgg16':
        #loading vgg16 as base model
        base_model = VGG16(input_shape=(64,64,3),include_top=False,weights="imagenet")
    elif model_name == 'resnet50':
        #loading resnet50 as base model
        base_model = ResNet50(input_shape=(64,64,3),include_top=False,weights="imagenet")
    elif model_name == 'inceptionv3':
        #loading inceptionv3 as base model
        base_model = InceptionV3(input_shape=(64,64,3),include_top=False,weights="imagenet")

    # Freezing base model Layers 
    for layer in base_model.layers:
        layer.trainable=False
        
    # Building our custom model layers on top of base model 
    model=Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(Dense(2,activation='softmax'))
    model.add(Dense(9,activation='sigmoid'))

    # defining f1_scroe mteric algorithm
    def f1_score(y_true, y_pred):
        true_positives_values = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positive_value = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives_value = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives_values / (predicted_positives_value + K.epsilon())
        recall = true_positives_values / (possible_positive_value + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    # Model metrics options
    METRICS = [BinaryAccuracy(name='accuracy'),Precision(name='precision'),Recall(name='recall'),AUC(name='auc'),f1_score]

    # Using this we can improve out effciency of traning
    # This primarily Reduces learning rate when a metric has stopped improving.

    reducelr_plateau = ReduceLROnPlateau(monitor = 'val_loss',patience = 5,verbose = 1,factor = 0.75, min_lr = 1e-10)

    # create a model checkpoint (save a model locally so to be used later.) 
    # A checkpoint works like a race checkpoint. Even if model has stoped training in the middle, one can continue traning from that checkpoint
    model_checkpoint = ModelCheckpoint(filepath=model_weights_path,save_freq='epoch',period=1)

    # This will stop training when a monitored metric has stopped improving
    early_stopping = EarlyStopping(verbose=1, patience=5)

    # Configuring our model
    model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=METRICS)

    # Training
    history=model.fit(X_train, y_train,validation_data=(X_test, y_test),verbose = 1,epochs = no_epochs,callbacks=[reducelr_plateau,model_checkpoint,early_stopping])

    # serialize model architecture to JSON
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")
    #%% PLOTTING RESULTS (Train vs Validation FOLDER 1)

    # Function to create model training stats
    def train_Validation_plot(acc,val_acc,loss,val_loss,auc,val_auc,precision,val_precision,f1,val_f1):
        fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize= (20,5))
        fig.suptitle("Visualization of Model")
        
        # Figure 1
        fig,ax = plt.subplots(figsize= (20,5))
        ax.plot(range(1, len(acc) + 1), acc)
        ax.plot(range(1, len(val_acc) + 1), val_acc)
        ax.set_title('Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend(['training', 'validation'])
        
        # Figure 2
        fig,ax = plt.subplots(figsize= (20,5))
        ax.plot(range(1, len(loss) + 1), loss)
        ax.plot(range(1, len(val_loss) + 1), val_loss)
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(['training', 'validation'])
        
        # Figure 3
        fig,ax = plt.subplots(figsize= (20,5))
        ax.plot(range(1, len(auc) + 1), auc)
        ax.plot(range(1, len(val_auc) + 1), val_auc)
        ax.set_title('AUC Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('AUC')
        ax.legend(['training', 'validation'])
        
        # Figure 4
        fig,ax = plt.subplots(figsize= (20,5))
        ax.plot(range(1, len(precision) + 1), precision)
        ax.plot(range(1, len(val_precision) + 1), val_precision)
        ax.set_title('Precision')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Precision')
        ax.legend(['training', 'validation'])
        
        # Figure 5
        fig,ax = plt.subplots(figsize= (20,5))
        ax.plot(range(1, len(f1) + 1), f1)
        ax.plot(range(1, len(val_f1) + 1), val_f1)
        ax.set_title('F1-score')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('F1 score')
        ax.legend(['training', 'validation'])
        
        plt.show()

    # Getting model stats Graph
    train_Validation_plot(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],history.history['auc'],history.history['val_auc'],history.history['precision'],history.history['val_precision'],history.history['f1_score'],history.history['val_f1_score'])
    final_model=model

    y_pred=[]
    i=0
    for img in X_val:
        img = image_reshape(img,image_mode)
        ynew = final_model.predict(img)
        pred = ynew.tolist()
        temp=unique_y[pred[0].index(max(pred[0]))]
        print(f"{i} = {temp}")
        y_pred.append(temp)
        i+=1


    def plot_confusion_matrix(cm,
                            target_names,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True):
        import matplotlib.pyplot as plt
        import numpy as np
        

        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(15, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig(confusion_matrix_path)

    print(type(y_val))
    print(len(y_pred))

    plot_confusion_matrix(cm=confusion_matrix(y_true=y_val, y_pred=y_pred), target_names=unique_label, normalize=False)

    def classification_report_txt(report):
        with open(classification_report_path, 'w') as f:
            f.write(report)

    report = classification_report(y_val, y_pred, target_names=unique_label)
    classification_report_txt(report)

for model_name in ['vgg16', 'resnet50', 'inceptionv3']:
    run_model(model_name)