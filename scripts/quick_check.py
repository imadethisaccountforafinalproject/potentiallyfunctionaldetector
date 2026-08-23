import os 
import sys

from PIL import Image as Img
  

IMAGE = "test_images/ai.jpeg"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detector  
image = Img.open(IMAGE)
image.load()
file_bytes = open(IMAGE, "rb").read()

result = detector.AI_WORDS

print("Image: ", IMAGE)
print("Result:", result)
