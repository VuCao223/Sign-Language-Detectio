import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


DATA_PATH = os.path.join('mp_Data')
actions = np.array(['Hello', 'How are you', 'Thanks', 'My name is', 'Bye', 'A', 'B', 'C', 'D','V','U'])
sequences, labels = [], []

no_sequences = 200
sequence_length = 30
label_map = {label:num for num, label in enumerate(actions)}

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])



X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=11)
# Chia tập train/validation từ tập train đã chia
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.125, shuffle=True,random_state=11)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
optimizer =Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
# -----------------------------MODEL LSTM--------------------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer= optimizer  , loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size = 32 ,shuffle = True,validation_split=0.1,callbacks = [early_stopping_callback])
# ---------------------------------------------------------------------------------
model.save('LSTMlan.h5')
model.summary()


# In thông tin về kích thước train/test/validation
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
print("Validation samples:", len(X_valid))

y_pred = model.predict(X_test)
predicted_labels = np.argmax(y_pred, axis=1)


# Tính toán chính xác cho từng lớp
class_accuracy = {}
for i, action in enumerate(actions):
    true_indices = np.where(np.argmax(y_test, axis=1) == i)[0]
    correct_predictions = np.sum(predicted_labels[true_indices] == i)
    total_samples = len(true_indices)
    class_accuracy[action] = correct_predictions / total_samples

# In kết quả chính xác cho từng lớp
for action, accuracy in class_accuracy.items():
    print(f"Class '{action}' Accuracy:", accuracy)

#----------------------------------------------------------------------------------
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print("Train Accuracy:", train_accuracy)
print("Train Loss:", train_loss)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
valid_loss, valid_accuracy = model.evaluate(X_valid, y_valid)
print("Validation Accuracy:", valid_accuracy)
print("Validation Loss:", valid_loss)


#---------------------------Model Accuracy on test set------------------------------------
test_true = np.argmax(y_test, axis=1)
test_pred = np.argmax(model.predict(X_test), axis=1)
print(" Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))





def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    epochs = range(len(metric_value_1))
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
    plt.title(str(plot_name))
    plt.legend()




#--------------------------------
y_hat = model.predict(X_test)
y_pred = np.argmax(y_hat, axis=1)
y_test_label =  np.argmax(y_test, axis=1)
# Tính accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_label, y_pred)
print('Accuracy: %f' % accuracy)
# Tính precision tp / (tp + fp)
precision = precision_score(y_test_label, y_pred, average='macro',zero_division=1)
print('Precision: %f' % precision)
# Tính recall: tp / (tp + fn)
recall = recall_score(y_test_label, y_pred, average='macro',zero_division=1)
print('Recall: %f' % recall)
# Tính f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_label, y_pred, average='macro')
print('F1 score: %f' % f1)
auc = roc_auc_score(y_test, y_hat, multi_class='ovr')
print('ROC AUC: %f' % auc)
print(model.evaluate(X_test, y_test))


#-----------Tính confusion matrix----------------------------------------------
pyplot.figure(figsize=(15,15))
matrix = confusion_matrix(y_test_label, y_pred)
labels = ['Hello', 'How are you', 'Thanks', 'My name is', 'Bye', 'A', 'B', 'C', 'D','V','U']
sns.heatmap(matrix, cmap="Blues" , annot = True, yticklabels=labels,xticklabels=labels,annot_kws={"fontsize": 16} )
pyplot.yticks(rotation= 0, fontsize=15)
pyplot.xticks(rotation= 90,fontsize=15)
pyplot.xlabel('Predictions', fontsize=18)
pyplot.ylabel('Actuals', fontsize=18)
pyplot.title('Confusion Matrix', fontsize=18)
print(matrix)


#-----------------------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
# plot accuracy during training
pyplot.figure(figsize=(20,10))
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['val_loss'], label='val_loss')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='acc')
pyplot.plot(history.history['val_accuracy'], label='val_acc')
pyplot.legend()
pyplot.show()









# Tính toán dự đoán của mô hình trên tập dữ liệu đánh giá
y_pred = model.predict(X_test)
predicted_labels = np.argmax(y_pred, axis=1)

# Tính toán độ chính xác toàn bộ tập dữ liệu
accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_labels)
print("Accuracy:", accuracy)


# In báo cáo đánh giá chi tiết


# Chuyển đổi target_names thành một danh sách
target_names = actions.tolist()

# Giả sử y_test và predicted_labels là các mảng numpy
classification_rep = classification_report(np.argmax(y_test, axis=1), predicted_labels, target_names=target_names)

# Tạo DataFrame từ báo cáo phân loại
report_df = pd.DataFrame(classification_report(np.argmax(y_test, axis=1), predicted_labels, target_names=target_names, output_dict=True)).transpose()

# In bảng thông tin phân loại
print("Classification Report:")
print(report_df)




# Vẽ biểu đồ cho accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.title('Loss')
plt.legend()
plt.show()








