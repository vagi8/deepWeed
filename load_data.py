
import cv2
import pickle
import pandas as pd
import os

def load_csv(path):
    df = pd.read_csv(path)
    return df

def read_image_array_data(data, df):
    image_arrays = []
    for image in os.listdir(data):
        if image.endswith('.jpg'):
            img = cv2.imread(data+image ,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (75,75), interpolation=cv2.INTER_NEAREST)
            image_arrays.append([img, df.loc[df['Filename']==image,'Label'].item(),df.loc[df['Filename']==image,'Species'].item()])
        else:
            print('not jpg - ', data+image)
        print(len(image_arrays))
    return image_arrays

def load_data_to_pickle(data, label,pickle_name):
    print("Making class0_array and class1_array ...")
    df=load_csv(label)
    idc_class_0_array = read_image_array_data(data,df)

    with open(pickle_name, 'wb') as output:
        pickle.dump(idc_class_0_array, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    label = '../deepWeed_dataset/weed_labels.csv'
    data = '../deepWeed_dataset/weed/'
    pickle_name = 'weed_train_array_75.pkl'
    load_data_to_pickle(data,label,pickle_name)


# print("Loading class0_array and class1_array ...")
# with open('./train_array.pkl', 'rb') as input:
#     idc_class_0_array = pickle.load(input)
# print(idc_class_0_array)