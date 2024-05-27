import tensorflow as tf

# Scarica il modello VGG16 pre-addestrato sui dati ImageNet
model = tf.keras.applications.MobileNet(
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# Salva il modello in un file
model.export('mobilenet.tf')
