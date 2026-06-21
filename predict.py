import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load the trained model
model = load_model('brain_tumor_model.keras')

# Class names
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load the test image
img = load_img('test.jpg', target_size=(256, 256))

# Convert image to array
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Prediction:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))