'''
This API is used for comparing images belonging to multiple classes and finding out defects. Steps are as follows:
  1. Upload the input image to the API. Along with it pass info about the type of image i.e grid or layout or power_eco, etc.,
  2. Image of the correponding type i.e not defective img will be fetched from local
  3. Siamese Network compares the difference between the images
  4a. If the difference/distance is very small then the input image should also be similar to local image 
      i.e the component in input image should also most probably not be defective
  4b. If the difference/distance is large then the input image is not similar to the local image
      i.e the component in input image should be defective
'''
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from pydantic import BaseModel, validator
#import tensorflow as tf
import os
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
  img = img[:, :, :3]
  img = img/255.0
  return img

def prediction(input_img_path, val_img_path, input_category):
  input_img = preprocess(input_img_path)
  val_img = preprocess(val_img_path)
  result = model.predict(list(np.expand_dims([input_img,val_img],axis=1)))
  print(result)
  if result>0.5:
    print(f"{input_category} switch component is not defective")
    return 'Not Defective'
  else:
    print(f"{input_category} switch component is defective")
    return 'Defective'


## Reload the model
## Not using the custom_objects will trigger errors
class L1Dist(Layer):
  def __init__(self, **kwargs):
    super().__init__()

  def call(self, input_embedding, validation_embedding):
    return abs(input_embedding-validation_embedding)
  
binary_cross_loss = BinaryCrossentropy()

model = load_model('multiple_classes_defect_detection_new.h5',
                    custom_objects={'L1Dist':L1Dist,'binary_cross_loss':BinaryCrossentropy})

app = FastAPI()

class ImageInput(BaseModel):
    file: UploadFile = File(...)
    input_text: str = Form(...)

    @validator("input_text")
    def validate_input_text(cls, value):
        allowed_values = {'fe','grid','four','eco_power','top','lights'}
        if value.lower() not in allowed_values:
            error = "Input should be one among the following ['fire_extinguisher','grid','layout','power_eco','top3','dashboard','dup','mic','two']"
            raise HTTPException(status_code=422, detail=str(error))
        return value.lower()


def perform_validation(image_input: ImageInput = Depends()):
    return image_input

@app.post("/upload/")
async def upload_image(image_input: ImageInput = Depends(perform_validation)):
    # Here, 'file' is the uploaded image.

    # You can save the uploaded file to a directory.
    with open(f"uploaded_images/{image_input.file.filename}", "wb") as f:
        f.write(image_input.file.file.read())

    input_img_path = f"uploaded_images/{image_input.file.filename}"

    val_img_type = image_input.input_text
    img_extension = ".jpeg"
    val_img_filename = val_img_type + img_extension

    val_img_path = f"multi_classes_images/{val_img_type}/"

    results = []
    for path in os.listdir(val_img_path):
      val_path = f"multi_classes_images/{val_img_type}/{path}"
      result = prediction(input_img_path, val_path, image_input.input_text)
      results.append(result)
    print(results)
    label = "Not Defective" if results.count("Not Defective") >= results.count("Defective") else "Defective" 
    
    #return {"filename": file.filename}
    return {"filename": image_input.file.filename, "category":image_input.input_text , "prediction":str(label)}
