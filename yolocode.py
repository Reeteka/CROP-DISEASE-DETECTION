import torch
import ultralytics
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from ultralytics import YOLO
import seaborn as sns
import pandas as pd
from PIL import Image

print("Torch version:", torch.__version__)
print("Ultralytics version:", ultralytics.__version__)

from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (you can select any model size)
model = YOLO('yolov8n.pt')  # 'n' for nano, you can use 's', 'm', 'l' for larger models

# Train the model using the dataset and data.yaml from Roboflow
model.train(data='C:/Users/vijay/OneDrive/Desktop/Mini_project/Apple.v1i.yolov8/data.yaml', epochs=20, imgsz=640, batch=16)


print(type(model))  # Ensure the model is a YOLO object
print(dir(model))   # List all attributes and methods for the object
print(model.names)   # List of names model contains


from PIL import Image
import matplotlib.pyplot as plt

# Load and display the training results image
img = Image.open("C:/Users/vijay/OneDrive/Desktop/Mini_project/runs/detect/train/results.png")  # Adjust path as needed
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()


# Save the trained model weights
best_model_path = 'C:/Users/vijay/OneDrive/Desktop/Mini_project/runs/detect/train/weights/best.pt'  # YOLOv8 will save the best weights during training
model.save(best_model_path)

print(f"Model saved at: {best_model_path}")



# Load the trained model
model = YOLO('C:/Users/vijay/OneDrive/Desktop/Mini_project/runs/detect/train/weights/best.pt')

# Evaluate the model on the validation set
metrics = model.val(data='C:/Users/vijay/OneDrive/Desktop/Mini_project/Apple.v1i.yolov8/data.yaml', imgsz=640)


# Access the results dictionary
metrics_dict = metrics.results_dict

# Output the evaluation metrics using the correct keys
print(f"Precision: {metrics_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {metrics_dict['metrics/recall(B)']:.4f}")
print(f"mAP@0.5: {metrics_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95: {metrics_dict['metrics/mAP50-95(B)']:.4f}")




# Assuming you have a dataframe with your class labels
data = {
    'classes': ['Cedar Apple Rust', 'Scab', 'Healthy', 'Black Rot'],  # Add all your classes here
    'counts': [14, 40, 15, 25]  # Replace with actual counts
}

df = pd.DataFrame(data)

plt.bar(df['classes'], df['counts'])
plt.xticks(rotation=45)
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Training Dataset')
plt.show()


results = model.predict(source='C:/Users/vijay/OneDrive/Desktop/Mini_project/scab.jpg')
print(results)  # To see the structure of the results

from PIL import Image
import matplotlib.pyplot as plt

# Load the image
img_path = 'C:/Users/vijay/OneDrive/Desktop/Mini_project/scab.jpg'
img = Image.open(img_path)

# Display the image using matplotlib
plt.imshow(img)
plt.axis('on')  # Hide axes
plt.show()


from ultralytics import YOLO

# Load the saved model weights for inference
trained_model = YOLO('C:/Users/vijay/OneDrive/Desktop/Mini_project/runs/detect/train/weights/best.pt')

# Use the model to make predictions on a new image
prediction_results = trained_model.predict(source='C:/Users/vijay/OneDrive/Desktop/Mini_project/scab.jpg', imgsz=640, conf=0.05)

# Iterate through the prediction results and plot them
for result in prediction_results:
    result.plot()
    plt.show() # This will plot each individual result
    


# Assume prediction_results contains bounding boxes and labels
boxes = prediction_results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
scores = prediction_results[0].boxes.conf.cpu().numpy()  # Confidence scores
labels = prediction_results[0].boxes.cls.cpu().numpy()  # Class labels

# Generate a heatmap of prediction confidence over the imag
if scores.size > 0:

  heatmap = sns.heatmap(scores.reshape(1, -1), annot=True, cmap='YlGnBu', cbar=True)
  plt.title("Confidence Scores for Detected Plants/Diseases")
  plt.show()
else:
  print("No objects detected")
  


# Load the saved model weights for inference
trained_model = YOLO('C:/Users/vijay/OneDrive/Desktop/Mini_project/runs/detect/train/weights/best.pt')

# Use the model to make predictions on a new image
img_path = 'C:/Users/vijay/OneDrive/Desktop/Mini_project/scab.jpg'
prediction_results = trained_model.predict(source=img_path, imgsz=640, conf=0.1)

# Check if any predictions were made
if len(prediction_results) > 0:
    for result in prediction_results:
        if result.boxes:  # Ensure there are boxes (detections) in the result
            # Display the image with predictions
            img_with_boxes = result.plot()  # result.plot() returns an image with predictions drawn on it

            # Convert the image (if necessary) to display using matplotlib
            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

            # Display the image using matplotlib
            plt.imshow(img_rgb)
            plt.axis('on')  # Hide axes
            plt.show()
        else:
            print("No objects detected in the image,Leaf is HEALTHY !!!")
else:
    print("No predictions were returned by the model.")
    
    
for result in prediction_results:
    boxes = result.boxes.xyxy.tolist()  # Extract bounding boxes
    scores = result.boxes.conf.tolist()  # Extract confidence scores
    labels = result.boxes.cls.tolist()  # Extract class labels


    label_names = [model.names[int(label)] for label in labels]

    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Labels:", label_names)


image_name = 'C:/Users/vijay/OneDrive/Desktop/Mini_project/scab.jpg'

# Check if the boxes, scores, and labels are being extracted properly
for result in prediction_results:
    print(result.boxes)  # Should display the bounding boxes
    print(result.boxes.xyxy)  # Bounding box coordinates
    print(result.boxes.conf)  # Confidence scores
    print(result.boxes.cls)  # Class labels

# Initialize an empty list to store predictions
predictions = []


# Log details for each prediction in the image
for box, score, label in zip(boxes, scores, labels):
    prediction_data = {
        'image': image_name,  # Add the image name
        'x1': box[0],         # x1 coordinate
        'y1': box[1],         # y1 coordinate
        'x2': box[2],         # x2 coordinate
        'y2': box[3],         # y2 coordinate
        'score': score,       # Confidence score
        'label': label_names        # Class label
    }
    print("Appending:", prediction_data)  # Print each row before appending
    predictions.append(prediction_data)

# Convert the predictions list to a DataFrame
df = pd.DataFrame(predictions)

# Define the CSV file path where the file will be saved
csv_file_path = 'C:/Users/vijay/OneDrive/Desktop/Mini_project/predictions_log.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"Predictions logged to {csv_file_path}")


df_check = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/Mini_project/predictions_log.csv")
print(df_check)
