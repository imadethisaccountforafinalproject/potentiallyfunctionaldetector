#from PIL import ImageChops, ImageFilter, ImageStat
#from PIL.ExifTags import TAGS


#AI_WORDS = ["Midjourney","stable diffusion","Stable diffusion","Stable Diffusion", "ComfyUI", "Gemini", "generated", "Generated", "gemini", "comfyui","midjourney","DALL-E","dall-e", "OpenAI", "openai", "Flux", "flux", "Sora", "sora"]
#AI_KEYS = ["parameters", "prompt", "workflow"]
#AI_SIZES = [(512,512),(1024,1024),(768,768),(832,1216),(1216,832),(1024,1792),(1792,1024)]


#from transformers import pipeline
#MODEL_NAME = "jacoballessio/ai-image-detect-distilled"
# return pipeline("image-classification", model=MODEL_NAME)
#def detect_image(detector, image):
# result = detector(image.convert("RGB"))[0]
# return result["label"], result["score"]



from transformers import pipeline
MODEL_NAME = "jacoballessio/ai-image-detect-distilled"
def load_detector():
 return pipeline("image-classification", model=MODEL_NAME)
def detect_image(detector, image):
 result = detector(image.convert("RGB"))[0]
 return result["label"], result["score"]
