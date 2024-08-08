import pyautogui
import time
import os

time.sleep(3)

# Obtener la ruta absoluta de la imagen
try:
    image_path = os.path.join(os.path.dirname(__file__), '../resources/edge_logo.png')
except NameError:
    image_path = os.path.join(os.getcwd(), 'resources/edge_logo.png')

print(f"Ruta de la imagen: {os.path.abspath(image_path)}")

# Usar la imagen con locateCenterOnScreen
image_loc = pyautogui.locateCenterOnScreen(image_path, grayscale=False, confidence=0.8)

if image_loc:
    pyautogui.click(image_loc)  # Corregido para usar las coordenadas encontradas
    print("Se encontró la imagen y se le dio click")
else:
    print("Imagen no encontrada")
    


