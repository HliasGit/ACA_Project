import tensorflow as tf

# Scarica il modello VGG16 pre-addestrato sui dati ImageNet
model = tf.keras.applications.MobileNet(
    weights="imagenet"
)

# Salva il modello in un file
model.save('mobileNet.h5')
