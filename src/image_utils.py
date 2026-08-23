from io import BytesIO as bio
from PIL import Image,ImageOps,UnidentifiedImageError 



MAX_FILE_SIZE_MB = 5
ALLOWED_FORMATS = {"png","jpeg","jpg"}


def prepare_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_size_MB = len(file_bytes) / (1024*1024)

    if file_size_MB > Max_FILE_SIZE_MB:
        return None, None, "Image must be smaller than 5 MB"
        