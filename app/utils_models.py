from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import RandomRotation
import tensorflow as tf
#tf.config.experimental.enable_op_determinism()


#from tensorflow.keras.config import enable_unsafe_deserialization

#enable_unsafe_deserialization()






# Charger les poids
#model.load_weights("./models/Brain_model_weights_2_simple.keras")

def predict_single(model, img, class_name):

    # Redimensionner l'image à (224, 224)
    img = img.resize((224, 224))  # Taille attendue par le modèle

    # Convertir l'image en tableau numpy et ajouter une dimension
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Ajouter une dimension (batch)

    # Faire des prédictions
    predictions = model.predict(img_array)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

import os
from django.conf import settings


def predict_image(request, model, class_name):
    print(request.FILES['image'])
    file = request.FILES['image']
    fs = FileSystemStorage()
    file_path_name = fs.save(file.name,file)
    print(file_path_name)
    file_path_name_ = fs.url(file_path_name)
    #file_path_name = "."+file_path_name_
    file_path_name = os.path.join(settings.MEDIA_ROOT, file_path_name)
    print(file_path_name)

    #file_path_name = "./media/Te-me_0017_pGnr9O7.jpg"

    img = tf.keras.utils.load_img(file_path_name)
    # Effectuer une prédiction et afficher le résultat
    predicted_class, confidence = predict_single(model, img, class_name)

    print(predicted_class, confidence)

    context = {'file_path_name':file_path_name_, 'predicted_class':predicted_class, 'confidence':confidence}
    return context









