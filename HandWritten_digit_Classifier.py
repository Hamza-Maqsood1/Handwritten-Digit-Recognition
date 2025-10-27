import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt


test_path = r"C:\Users\Hamza\Downloads\Cloud_credits\Hand Written Digit Classifier\Reduced MNIST Data\Reduced Testing data"
train_path = r"C:\Users\Hamza\Downloads\Cloud_credits\Hand Written Digit Classifier\Reduced MNIST Data\Reduced Trainging data"

train_datagen = ImageDataGenerator(1./255)
test_datagen=ImageDataGenerator(1./255)

train_data=train_datagen.flow_from_directory(
    train_path,
    color_mode="grayscale",
    class_mode="categorical",
    target_size=(28, 28),
    batch_size=32,
    shuffle = True
)

test_data=test_datagen.flow_from_directory(
    test_path,
    class_mode="categorical",
    color_mode="grayscale",
    target_size=(28, 28),
    batch_size=32,
    shuffle=False 
)


model=Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64,(3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history=model.fit(
    train_data,
    epochs=20,
    validation_data=test_data
)

test_loss, test_acc=model.evaluate(test_data)
print("Test Accuracy : ", test_acc)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("mnist_cnn_model.h5")




import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_prob = model.predict(test_data)
y_pred_classes = np.argmax(y_pred_prob, axis=1)
y_true = test_data.classes


cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(range(10)),
            yticklabels=list(range(10)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Handwritten Digit Recognition')
plt.show()


print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes))


x_test_batch, y_test_batch = next(test_data)

predictions = model.predict(x_test_batch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_batch, axis=1)

plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test_batch[i].reshape(28,28), cmap='gray')
    plt.title(f"True: {true_classes[i]}, Pred: {predicted_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
