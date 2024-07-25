# actions.py

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# Load the pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Preprocess input image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:  # remove alpha channel if present
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(image_array)

# Decode the predictions
def decode_predictions(predictions: np.ndarray) -> List[str]:
    decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=3)
    return [f"{pred[1]}: {pred[2]*100:.2f}%" for pred in decoded_predictions[0]]

class ActionCheckSymptoms(Action):
    def name(self) -> Text:
        return "action_check_symptoms"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        symptoms = tracker.latest_message['text']
        dispatcher.utter_message(text=f"Got it. You mentioned the following symptoms: {symptoms}")
        return [SlotSet("symptom", symptoms)]

class ActionAnalyzeImage(Action):
    def name(self) -> Text:
        return "action_analyze_image"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        image_url = tracker.latest_message['text']
        
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image)
            decoded_predictions = decode_predictions(predictions)
            
            diagnosis = "\n".join(decoded_predictions)
            dispatcher.utter_message(text=f"Analysis complete. Here are the top predictions:\n{diagnosis}")
            
        except Exception as e:
            dispatcher.utter_message(text=f"Failed to analyze the image. Error: {str(e)}")
        
        return [SlotSet("image", image_url)]
