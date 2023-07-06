from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch
import cv2
import time

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model = model.eval()

process = weights.transforms(antialias=True)

img = cv2.imread("bird.jpeg")
# Convert image from BGR to RGB.
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert image from numpy.ndarray to torchvision image format.
rgb_image = rgb_image.transpose((2, 0, 1))
rgb_image = rgb_image / 255.0
rgb_image = torch.FloatTensor(rgb_image)

batch = process(rgb_image).unsqueeze(0)

# Step 4: Use the model and print the predicted category
startTime = time.time()
prediction = model(batch).squeeze(0).softmax(0)
print(f"Time: {time.time() - startTime:.3f}s")
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
