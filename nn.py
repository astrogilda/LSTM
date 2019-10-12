
# to suppress future warning from tf + np 1.17 combination.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import sys
import resource

epsilon = 1e-5

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemAvailable:':
                free_memory = int(sline[1])
                break
            #if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
            #    free_memory += int(sline[1])
    return free_memory


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 1.5, hard))

#memory_limit()

# libraries for read in data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations
import matplotlib.pyplot as plt
# %matplotlib inline

# libraries needed for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer as pt, StandardScaler as ss, MinMaxScaler as mms, RobustScaler as rs

import tensorflow as tf
from keras.models import Sequential, load_model
#from keras.layers import Dense, Activation, Dropout, TimeDistributed, LSTM, Flatten, Bidirectional
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
from keras import backend as K
# The bottomline is that you must stick to functions
# from keras.backend in all your functions. If you are not keen on compatibility
#(i.e. you only intend to run your models on a particular backend, be it Theano or Tensorflow),
# you can also use the operations defined by your backend directly,
# because keras.backend is relatively limited.
# Remember than Keras needs a symbolic tensor for its loss function, but numpy.functions()
# return non-symbolic values, therefore, if you define your custome loss or metric functions,
# you shall stick to keras.backend

from sklearn.model_selection import StratifiedKFold

# all features
feature_names = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE', 'XR_MAX']
# relevent features by Fischer ranking score and manually looking at the histograms in positive and negative classes.
# I choose those features with reletively high Fischer ranking score & those whose histograms look different in
# positive and negative classes.
# I further drop features according to their physical definitions. When some features' definitions are related to each
# other, I first double check their correlation by looking at their scatter plot. If their correlation is confirmed visually,
# I drop n number features from the correlated features if there are n confirmed correlations.
relevant_features_0 = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD', 'R_VALUE']

# By observing the histograms of relevant features, their histograms can be grouped into four categories.
# right skewed with extreme outliers, right skewed without extreme outliers, left skewed with extreme outliers, non skewed
right_skewed_features = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'EPSZ', 'MEANSHR', 'MEANGAM', 'MEANGBH', 'MEANJZD']
right_skewed_features_with_ol = ['TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'SAVNCPP', 'USFLUX', 'MEANSHR', 'MEANGAM', 'MEANGBH', 'MEANJZD']
right_skewed_features_without_ol = ['TOTUSJH', 'ABSNJZH', 'EPSZ']
left_skewed_features_with_ol = ['TOTFZ']
non_skewed_features = ['MEANGBT', 'R_VALUE']

# I further decide that TOTFZ is correlated with EPSZ and TOTBSQ. Furthermore, TOTFZ cannot be well scaled yet, I
# decide to drop it for now. Note that TOTFZ is the only feature in the list `left_skewed_with_ol`. In the end, I select
# 14 features for fitting the data, their names are stored in the list called `selected_features`.
selected_features = right_skewed_features + non_skewed_features
### selecting ALL features, to see how this changes performance
selected_features = feature_names

print('{} are selected for training.'.format(len(selected_features)))
print('selected features include \n',selected_features)

# get the indice for features
indice_right_skewed_with_ol = []
indice_right_skewed_without_ol = []
indice_non_skewed = []
for i in range(0,len(selected_features)):
    if selected_features[i] in right_skewed_features_with_ol:
        indice_right_skewed_with_ol.append(i)
    elif selected_features[i] in right_skewed_features_without_ol:
        indice_right_skewed_without_ol.append(i)
    elif selected_features[i] in non_skewed_features:
        indice_non_skewed.append(i)


scale_params_right_skewed = pd.read_csv('scale_params_right_skewed.csv')
scale_params_right_skewed.set_index('Unnamed: 0', inplace=True)

scale_params_non_skewed = pd.read_csv('scale_params_non_skewed.csv')
scale_params_non_skewed.set_index('Unnamed: 0', inplace=True)

# Functions for reading in data from .json files
def decode_obj(line, pos=0, decoder=JSONDecoder()):
    no_white_space_regex = re.compile(r'[^\s]')
    while True:
        match = no_white_space_regex.search(line, pos)
        # line is a long string with data type `str`
        if not match:
            # if the line is full of white space, get out of this func
            return
        # pos will be the position for the first non-white-space character in the `line`.
        pos = match.start()
        try:
            # JSONDecoder().raw_decode(line,pos) would return a tuple (obj, pos)
            # obj is a dict, and pos is an int
            # not sure how the pos is counted in this case, but it is not used anyway.
            obj, pos = decoder.raw_decode(line, pos)
            # obj = {'id': 1, 'classNum': 1, 'values',feature_dic}
            # here feature_dic is a dict with all features.
            # its key is feature name as a str
            # its value is a dict {"0": float, ..., "59": float}
        except JSONDecodeError as err:
            print('Oops! something went wrong. Error: {}'.format(err))
            # read about difference between yield and return
            # with `yield`, obj won't be assigned until it is used
            # Once it is used, it is forgotten.
        yield obj

def get_obj_with_last_n_val(line, n):
    # since decode_obj(line) is a generator
    # next(generator) would execute this generator and returns its content
    obj = next(decode_obj(line))  # type:dict
    id = obj['id']
    class_label = obj['classNum']
    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    last_n_indices = np.arange(0, 60)[-n:]
    data = data.loc[last_n_indices]
    return {'id': id, 'classType': class_label, 'values': data}

def convert_json_data_to_nparray(data_dir: str, file_name: str, features):
    """
    Generates a dataframe by concatenating the last values of each
    multi-variate time series. This method is designed as an example
    to show how a json object can be converted into a csv file.
    :param data_dir: the path to the data directory.
    :param file_name: name of the file to be read, with the extension.
    :return: the generated dataframe.
    """
    fname = os.path.join(data_dir, file_name)
    all_df, labels, ids = [], [], []
    with open(fname, 'r') as infile: # Open the file for reading
        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).
            obj = get_obj_with_last_n_val(line, 60) # obj is a dictionary
            # if the classType in the sample is NaN, we do not read in this sample
            if np.isnan(obj['classType']):
                pass
            else:
                # a pd.DataFrame with shape = time_steps x number of features
                # here time_steps = 60, and # of features are the length of the list `features`.
                df_selected_features = obj['values'][features]
                # a list of np.array, each has shape=time_steps x number of features
                # I use DataFrame here so that the feature name is contained, which we need later for
                # scaling features.
                all_df.append(np.array(df_selected_features))
                labels.append(obj['classType']) # list of integers, each integer is either 1 or 0
                ids.append(obj['id']) # list of integers
    return all_df, labels, ids

print('Files contained in the ../input directiory include:')
print(os.listdir("./input"))

path_to_data = "./input"
file_name = "fold3Training.json"
fname = os.path.join(path_to_data,file_name)

# Read in all data in a single file
all_input, labels, ids = convert_json_data_to_nparray(path_to_data, file_name, selected_features)

# Change X and y to numpy.array in the correct shape.
X = np.array(all_input)
y = np.array([labels]).T
print("The shape of X is (sample_size x time_steps x feature_num) = {}.".format(X.shape))
print("the shape of y is (sample_size x 1) = {}, because it is a binary classification.".format(y.shape))

# define a function for scaling X
def scale_features(X, selected_features):
    X_copy = X.copy() # make a copy of X, must use np.array.copy(), otherwise if use X_copy = X, X_copy would point to the same memory, and once X or X_copy gets changed, both will change.
    for i in range(0,len(selected_features)):
        feature = selected_features[i] # str, feature name
        # right skewed with extreme outliers
        if feature in right_skewed_features_with_ol:
            x_min, y_median, y_IQR = scale_params_right_skewed.loc[['x_min','y_median','y_IQR'], feature]
            x = X[:,:,i] # n_sample x time_steps x 1
            y = np.log(x - x_min + 1.0)
            z = (y - y_median)/y_IQR
            X_copy[:,:,i] = np.nan_to_num(z)
        # right skewed without extreme outliers
        elif feature in right_skewed_features_without_ol:
            x_min, y_mean, y_std = scale_params_right_skewed.loc[['x_min','y_mean','y_std'], feature]
            x = X[:,:,i]
            y = np.log(x-x_min+1.0)
            z = (y - y_mean)/y_std
            X_copy[:,:,i] = np.nan_to_num(z)
        # non_skewed features, they do not have extreme outliers
        elif feature in non_skewed_features:
            x_mean, x_std = scale_params_non_skewed.loc[['x_mean','x_std'],feature]
            x = X[:,:,i]
            X_copy[:,:,i] = np.nan_to_num((x - x_mean)/x_std)
        else:
            print(feature+' is not found, and thus not scaled.')
    return X_copy

# Scale X
#X_scaled = scale_features(X, selected_features)
X_scaled = np.zeros_like(X)
for i in range(X_scaled.shape[1]):
    X_scaled[:,i] = pd.DataFrame(X[:,i]).interpolate(method='cubic', limit_direction='both', axis=0).values



def timeseries_scaling(X, is_training_data=True, list_of_transformers=None):
    #X += epsilon
    #X.shape = (nsamples, timesteps, features) 
    #is_training_data and list_of_transformers can take values 'True' and 'None, and 'False' and 'not None' only.
    X_new = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_new[i] = mms().fit_transform(X[i])
    #
    #"""
    X_new2 = X_new.copy()
    if is_training_data:
        list_of_transformers = list()
    #
    for i in range(X.shape[1]):
        if is_training_data:
            tr = ss()
            tr.fit(X_new[:,i])
            list_of_transformers.append(tr)
        else:
            tr = list_of_transformers[i]
        X_new2[:,i] = tr.transform(X_new[:,i])       
    return X_new2, list_of_transformers
    #"""
    #return X_new



# check NaN in y, X, X_scaled
print('There are {} NaN in y.'.format(np.isnan(y).sum()))
print('There are {} NaN in X.'.format(np.isnan(X).sum()))
print('There are {} NaN in X_scaled.'.format(np.isnan(X_scaled).sum()))


# Set some hyperparameters
n_sample = len(y)
time_steps = 60
batch_size = 64
feature_num = len(selected_features) # 25 features per time step
hidden_size = feature_num
use_dropout = True
use_callback = False # to be added later



#'Class Balanced Loss Based on Effective Number of Samples
unique_targets, unique_targets_cnts = np.unique(y, return_counts=True)
y_cnts = np.zeros_like(y)
for i,j in zip(unique_targets,unique_targets_cnts):
    y_cnts += (y==i)*j

y_cnts = y_cnts/len(y)

beta_cb = 0.999
alpha_cb = (1-beta_cb)/(1-beta_cb**y_cnts)
alpha_cb_norm_fac = len(unique_targets)/np.sum(np.unique(alpha_cb))
alpha_cb *= alpha_cb_norm_fac


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)
y_dim = np.shape(y)[1] # y=0 if no flare, y=1 if flare


# Split X, y into training and validation sets
"""
val_fraction = 0.33
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=val_fraction, random_state=0)
"""
# define 10-fold cross validation test harness
seed = 10
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []


# Define metric, which does not depend on imbalance of positive and negative classes in validation/test set
# Defining sensitivity = true_positive/(total real positive) = tp/(tp+fn)
# sensitivity is the same as recall
def sensitivity(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)
    true_positives = K.sum(K.round(y_true * y_pred)) 
    # K.clip(x,a,b) x is a tensor, a and b are numbers, clip converts any element of x falling
    # below the range [a,b] to a, and any element of x falling above the range [a,b] to b.
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # K.epsilon >0 & <<< 1, in order to avoid division by zero.
    sen = recall = true_positives / (possible_positives + K.epsilon())
    return sen

# Specificity = true_negative/(total real negative) = tn/(tn+fp)
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    spec = true_negatives / (possible_negatives + K.epsilon())
    return spec

# Precision = true_positives/predicted_positives = tp/(tp+fp)
def precision(y_true, y_pred):
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    true_positives = K.sum(K.round(y_true * y_pred)) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = true_positives / (predicted_positives + K.epsilon())
    return prec

# Informedness = sensitivity + specificity - 1
def informedness(y_true, y_pred):
    return sensitivity(y_true, y_pred)+specificity(y_true, y_pred)-1

# f1 = 2/((1/precision) + (1/recall))
def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    sen = sensitivity(y_true, y_pred)
    f1 = 2*((prec*sen)/(prec + sen + K.epsilon()))
    return f1



alpha = 0.5
gamma = 2

# alpha and gamma are to be set by CV

def weighted_cross_entropy(targets, inputs, alpha=alpha, gamma=gamma, alpha_cb=alpha_cb):
    # we use a modulating factor which down-weights the loss assigned to well-classified examples to prevent numerous easy examples from overwhelming the classifier.    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    bce = K.binary_crossentropy(targets, inputs)
    bce_exp = K.exp(-bce)
    # focal loss
    fl = K.mean(alpha_cb * K.pow((1-bce_exp), gamma) * bce)
    # soft focal loss - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8725513
    M_a = K.pow((1-bce_exp), (gamma*bce_exp))
    M_r = K.exp(1-bce_exp)/K.sum(K.exp(1-bce_exp))
    sfl = K.mean(alpha_cb * M_a * M_r * bce)
    return sfl


def mish(x):
    return x*K.tanh(K.softplus(x))

# Build LSTM networks using keras
num_folds = 5
num_epochs = 50

def fit_model(X=X_scaled, y=y, batch_size=batch_size, n_splits = num_folds, epochs=num_epochs):
    """
    # Calculate the imbalance between positive and negative classes in the data set
    n_tot = len(y_train)
    n_P = y_train.flatten().sum()
    n_N = n_tot - n_P
    print('There are {} positive-class samples.'.format(n_P))
    print(' and {} negative-class samples.'.format(n_N))
    # Set weights for positive and negative classes. Larger weight means more penalty if misclassifying
    N_weight = 1 # treat one negative training sample as N_weight=1 sample.
    P_weight = round(n_N/n_P*0.5) # (num_zero/num_one)*t, t=1 if you pays all attention to sensitivity (recall) = TP/P.
    # t=0 if one pays all attention to specificity = TN/N. t=0.5 if you care equally about sensitivity and specificity.
    #Here num_zero = number of zeros (negatives/no flares) in the training data.
    # num_one = number of ones (positives/flares) in the training data.
    # treat one positive training sample as P_weight=50 samples.
    # class_weight is applied onto the loss function during the training process
    print('weight for negative class is {}.'.format(N_weight))
    print('weight for positive class is {}.'.format(P_weight))
    """
    ###############
    ###############
    seed = 10
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(np.asarray(labels), np.asarray(labels)):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        #X_train = timeseries_scaling(X_train)
        #X_test = timeseries_scaling(X_test)
        #"""
        X_train, X_train_transformers = timeseries_scaling(X_train)
        X_test, _ = timeseries_scaling(X_test, is_training_data=False, list_of_transformers=X_train_transformers)
        #"""
        #####
        model = Sequential()
        # first LSTM layer, input_shape = (batch_size, time_steps, feature_num),
        # output_size = (batch_size, time_steps, hidden_size)
        # the first dimension does not need to be specified (batch_size)
        # the second dimension of the output is 1 if return_sequences = False, is time_steps if return_sequences = True
        model.add(Bidirectional(LSTM(units=hidden_size, input_shape=(time_steps,feature_num), return_sequences=True), merge_mode = 'ave'))
        # second LSTM layer, input_shape = (batch_size, time_steps, feature_num)
        # output_shape = (batch_size, 1, feature_num)
        model.add(Bidirectional(LSTM(units=hidden_size, return_sequences=True), merge_mode = 'ave'))
        if use_dropout:
            model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(int(hidden_size/2), activation=mish)))
        model.add(Flatten())
        # we add the Dense layer and the Activation layer separately, so we can access to the output of the Dense layer
        # which is the linear combination = weight*input + bias, before going through the nonlinear activation function
        model.add(Dense(y_dim)) # Dense layer has y_dim=1 or 2 neuron.
        # dropout to avoid overfitting if use_dropout
        model.add(Activation('softmax'))
        # Compile the model
        model.compile(loss=weighted_cross_entropy,optimizer='adam', metrics=[f1_score])
        model.fit(x=X_train, y=y_train, epochs=epochs, verbose=2, batch_size=batch_size, validation_data=(X_test, y_test))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score, informedness])
        # print the summary of the model
        # print(model.summary())
        #model.fit(X[train], y[train], epochs=10, verbose=2, batch_size=batch_size, class_weight={0:N_weight,1:P_weight})
       	# evaluate the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    return model

fit_model()






"""
# simple ensembling - mean averaging with equal model weights
def ensemble_predictions(members, X_val):
    # make predictions
    yhats = [model.predict(X_val) for model in members]
    yhats = array(yhats)
    # sum across ensemble members
    summed = numpy.sum(yhats, axis=0)
    # argmax across classes
    result = argmax(summed, axis=1)
    return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(X_val, y_val, members, n_members=10):
    # select a subset of members
    subset = members[:n_members]
    print(len(subset))
    # make prediction
    yhat = ensemble_predictions(subset, X_val)
    # calculate accuracy
    return accuracy_score(y_val, yhat)


# fit all models
n_members = 1
members = [fit_model(X_train, y_train) for _ in range(n_members)]
# evaluate different numbers of ensembles
scores = list()
for i in range(1, n_members+1):
    score = evaluate_n_members(X_val, y_val, members, i)
    print('> %.3f' % score)
    scores.append(score)

# plot score vs number of ensemble members
x_axis = [i for i in range(1, n_members+1)]
plt.plot(x_axis, scores)
plt.show()

"""







"""
# train the network, note the argument class_weight, which will put more penalty for misclassifying minor class
#history = model.fit(X_train, y_train, batch_size=batch_size, epochs=10, class_weight={0:N_weight,1:P_weight}, validation_data=(X_val,y_val))

# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, val_acc = model.evaluate(X_val, y_val, verbose=0)
print('Train: %.3f, Val: %.3f' % (train_acc, val_acc))


# plot history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
"""
