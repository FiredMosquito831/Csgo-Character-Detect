import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Function to perform object detection using YOLOv8
def detect_objects(image_path, model_path, save_results=True):
    # Load your custom YOLOv8 model
    model = YOLO(model_path).to(torch.device(0 if torch.cuda.is_available() else 'cpu'))

    # Run object detection on the image
    results = model(image_path)

    # Display detection results in the console
    print(results)

    # Save the results (image with bounding boxes) if save_results is True
    if save_results:
        # Get the directory where the input image is located
        image_dir = os.path.dirname(image_path)
        # Create a filename for the output image (with detections)
        output_image_path = os.path.join(image_dir, 'detected_' + os.path.basename(image_path))

        # Plot and save the image manually
        for result in results:
            # Get the original image
            img_with_boxes = result.plot()  # This returns an image with bounding boxes drawn
            cv2.imwrite(output_image_path, img_with_boxes)  # Save the output image

        print(f"Results saved to {output_image_path}")

    # Optionally return the results
    return results

# Function to display image with bounding boxes using OpenCV and Matplotlib
def display_image_with_boxes(image_path):
    # Load the result image from the same folder as input
    image_dir = os.path.dirname(image_path)
    result_img_path = os.path.join(image_dir, 'detected_' + os.path.basename(image_path))

    if os.path.exists(result_img_path):
        # Load and display the image with bounding boxes drawn
        img = cv2.imread(result_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis("off")  # Hide axes
        plt.show()
    else:
        print(f"Result image not found at {result_img_path}")

# Main function
if __name__ == "__main__":
    for i in range (1, 12, 1):
        # Path to the image you want to analyze
        test = f'test{str(i)}.png'
        image_path = f"C:/Users/fgghk/OneDrive/Desktop/{test}"  # Replace with your image path

        # Path to your custom YOLOv8 model
        model_path = 'C:\\Users\\fgghk\\PycharmProjects\\TranscriptionAppRelease\\CsgoAiDetectCharacters\\runs\\CurrentBEST\\train2\\weights\\best.pt'  # Replace with your YOLOv8 model path

        # Run object detection
        results = detect_objects(image_path, model_path)

        # Display the image with bounding boxes
        display_image_with_boxes(image_path)
