import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import to_categorical
import pickle
from sklearn.metrics import classification_report
import csv
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


dataset = pd.read_csv('fruit_dataset.csv')

X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

dummy_y = label_binarize(y_train, classes=[0, 1, 2, 3])
n_classes = dummy_y.shape[1]

dummy_y1 = label_binarize(y_test, classes=[0, 1, 2, 3])


classifier = Sequential()
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 5))
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))

sgd = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = classifier.fit(X_train, dummy_y, validation_split=0.33, batch_size = 10, nb_epoch = 1000, verbose=1)

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_test)

dfObj = pd.DataFrame(y_pred, columns=list('0123'))
maxValueIndexObj = dfObj.idxmax(axis=1)

lol = maxValueIndexObj.values.tolist()
lol1 = np.transpose(lol)
results = list(map(int, lol1))
	
cm = confusion_matrix(y_test, results, labels=[0, 1, 2, 3])
print(cm)

target_names = ['class 0', 'class 1', 'class 2', 'class 3']
print(classification_report(y_test, results, target_names=target_names))
report = classification_report(y_test, results, target_names=target_names)

fname = 'classification_report.csv'
with open(fname, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(report)
    writer.writerow(cm)



lw = 1.5

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(dummy_y1[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(dummy_y1.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0, 1.0])
plt.ylim([0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC.png', dpi=300)
plt.show()

plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1.05)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_extend.png', dpi=300)
plt.show()

precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(dummy_y1[:, i], y_pred[:, i])                                                    
    average_precision[i] = average_precision_score(dummy_y1[:, i], y_pred[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(dummy_y1.ravel(), y_pred.ravel())  
average_precision["micro"] = average_precision_score(dummy_y1, y_pred, average="micro")
                                                     
plt.plot(recall[0], precision[0], label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average Precision-Recall curve: AUC={0:0.2f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.savefig('Precision.png', dpi=300)
plt.show()

plt.plot(recall["micro"], precision["micro"],
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i in range(n_classes):
    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for all classes')
plt.legend(loc="lower right")
plt.savefig('precision_average.png', dpi=300)
plt.show()


