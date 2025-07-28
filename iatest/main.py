import tensorflow as tf
import numpy as np
import json

# Load the saved model
loaded_model = tf.keras.models.load_model('month_predictor.keras')

# Load the index_to_label dictionary from the json file
with open('index_to_label.json', 'r') as f:
    index_to_label = json.load(f)

# The keys will be loaded as strings
index_to_label = {int(k): v for k, v in index_to_label.items()}


# Use the loaded model to make predictions
new_months = ["December", "September", "January"]
new_months_dataset = tf.data.Dataset.from_tensor_slices(new_months)
predictions = loaded_model.predict(new_months_dataset.batch(1))

# Get the predicted month for each input month
for i, month in enumerate(new_months):
  predicted_index = np.argmax(predictions[i])
  predicted_month = index_to_label[predicted_index]
  print(f"The model predicts that the month after {month} is: {predicted_month}")
