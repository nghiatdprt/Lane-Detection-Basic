import project_config
import pickle
import os
import cv2
import logging
import numpy as np
def load_data():
    def load_label():
        path = os.path.join(project_config.dataset, 'infos.pkl')
        infile = open(path, 'rb')
        data = pickle.load(infile)
        infile.close()
        data = data['images']
        label = {}
        for ele in data:
            record = data[ele]
            label[record['output_name']] = record['label']
        return label
    def load_image():
        images = {}
        cwd = os.getcwd()
        os.chdir(project_config.dataset)
        for img in os.listdir(project_config.dataset):
            abs_path = os.path.abspath(img)
            if abs_path.endswith('.jpg'):
                name = os.path.splitext(os.path.basename(abs_path))[0]
                images[name] = cv2.imread(abs_path)
        os.chdir(cwd)
        return images
    labels = load_label()
    images = load_image()
    data = []
    for name in images:
        data.append((images[name], labels[name]))
    logging.info("Load dataset successfully. Dataset size: {}".format(len(data)))
    return data

def save_data_to_pkl(dataset, path):
    abs_path = os.path.abspath(path)
    outfile = open(abs_path, "wb")
    pickle.dump(dataset, outfile)
    outfile.close()
    logging.info("Dataset has been saved successfully on {}".format(abs_path))

def load_data_from_pkl(path):
    abs_path = os.path.abspath(path)
    infile = open(abs_path, "rb")
    dataset = pickle.load(infile)
    infile.close()
    logging.info("Dataset has been loaded successfully")
    return dataset

def extract_dataset(dataset):
    return list(zip(*dataset))

def load_toulouse_dataset():
    dataset = load_data_from_pkl("dataset.pkl")
    dataset_size = len(dataset)
    border = int(dataset_size*(1)-1)
    training_set = dataset[0:border]
    test_set = dataset[border:dataset_size]

    x_train, y_train = extract_dataset(training_set)
    x_test, y_test = extract_dataset(test_set)

    return (np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))