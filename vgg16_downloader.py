import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Scarica il modello VGG16 pre-addestrato sui dati ImageNet
model = VGG16(weights='imagenet')

# Salva il modello in un file
model.save('vgg16.h5')

print("Modello VGG16 scaricato e salvato come vgg16.h5")