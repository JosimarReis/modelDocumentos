import sys
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# todas as categorias
CATEGORIES = ["rg", "cpf", "cnh", "certidao", "ctps", "endereco", "rejeitar"]


def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = load_model('CNN.model')
nome = sys.argv[1]
image_string = sys.argv[1]  # os.path.join('testes', nome)
image = prepare(image_string)


def predict(image):
    prediction = model.predict([image], use_multiprocessing=True)[0]
    prediction = list(prediction)
    print(CATEGORIES[prediction.index(max(prediction))])
    sys.stdout.flush()


predict(image)
