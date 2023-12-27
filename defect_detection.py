'''
This API is used for comparing clamp images and finding out defects. Steps are as follows:
  1. Upload a clamp image to the API
  2. Image of a perfect clamp i.e not defective clamp will be fetched from local
  3. Siamese Network compares the difference between the images
  4a. If the difference/distance is very small then the input image should also be similar to local image 
      i.e clamp in input image should also most probably not be defective
  4b. If the difference/distance is large then the input image is not similar to the local image
      i.e clamp in input image should be defective
'''
from fastapi import FastAPI, File, UploadFile, Form
#import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.io import read_file,decode_jpeg
from tensorflow.image import resize
from tensorflow.random import uniform
from tensorflow.errors import InvalidArgumentError
from tensorflow.math import abs

def preprocess(file_path):
  byte_img = read_file(file_path)
  try:
    img = decode_jpeg(byte_img)
  except InvalidArgumentError:
    img = uniform([250, 250, 3], minval=0, maxval=256)
  img = resize(img,(100,100))
  img = img/255.0
  return img

def prediction(input_img_path,val_img_path):
  input_img = preprocess(input_img_path)
  val_img = preprocess(val_img_path)
  result = model.predict(list(np.expand_dims([input_img,val_img],axis=1)))
  print(result)
  if result>0.5:
    print('Clamp is not misplaced')
    return 'Not Misplaced'
  else:
    print('Clamp is misplaced')
    return 'Misplaced'


## Reload the model
## Not using the custom_objects will trigger errors
class L1Dist(Layer):
  def __init__(self, **kwargs):
    super().__init__()

  def call(self, input_embedding, validation_embedding):
    return abs(input_embedding-validation_embedding)
  
binary_cross_loss = BinaryCrossentropy()

model = load_model('clamp_defect_detection.h5',
                    custom_objects={'L1Dist':L1Dist,'binary_cross_loss':BinaryCrossentropy})

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile):
    # Here, 'file' is the uploaded image.

    # You can save the uploaded file to a directory.
    with open(f"uploaded_images/{file.filename}", "wb") as f:
        f.write(file.file.read())

    input_img_path = f"uploaded_images/{file.filename}"
    val_img_path = "1.jpeg"
    result = prediction(input_img_path,val_img_path)
    #return {"filename": file.filename}
    return {"filename": file.filename, "prediction":str(result)}
