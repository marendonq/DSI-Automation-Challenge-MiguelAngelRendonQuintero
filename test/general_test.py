import pyautogui
import os

# Obtener la ruta absoluta de la imagen
image_path = os.path.join(os.path.dirname(__file__), '../resources/edge_logo.png')
print(f"Ruta de la imagen: {os.path.abspath(image_path)}")

# Verificar si el archivo existe en la ruta especificada
if os.path.exists(image_path):
    print("El archivo existe en la ruta especificada.")
else:
    print("El archivo no se encuentra en la ruta especificada.")
