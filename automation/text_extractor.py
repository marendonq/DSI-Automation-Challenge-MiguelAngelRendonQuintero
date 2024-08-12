import os

def resolve_path(relative_path: str) -> str:
    """
    Resuelve una ruta relativa a una ruta absoluta basada en el archivo actual o el directorio de trabajo actual.

    Parámetros:
    -----------
    relative_path : str
        La ruta relativa del archivo.

    Retorno:
    --------
    str
        La ruta absoluta del archivo.
    """
    # Verificar si la ruta es ya absoluta
    if os.path.isabs(relative_path):
        return relative_path
    
    # Resolver la ruta basada en la ubicación del archivo actual o el directorio de trabajo
    base_path = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    absolute_path = os.path.join(base_path, relative_path)

    return os.path.abspath(absolute_path)

import pytesseract
from PIL import Image
from utils.image_preprocessing import preprocess_image

image_path = resolve_path("resources/final_processed_image.png")
preprocessed_image = preprocess_image(image_path)

img = Image.open(image_path)
ocr_result = pytesseract.image_to_string(img)

print(ocr_result)