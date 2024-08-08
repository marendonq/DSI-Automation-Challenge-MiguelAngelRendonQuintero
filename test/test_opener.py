#Verificar permisos sobre la pantalla
"""
import pyautogui

try:
    screenshot = pyautogui.screenshot()
    screenshot.show()
    print("Captura de pantalla exitosa")
except Exception as e:
    print(f"Error al tomar la captura de pantalla: {e}")
"""

#Verificar que si pueda localizar una imagen

"""
import pyautogui
import os

# Obtener la ruta absoluta de la imagen
image_path = os.path.join(os.path.dirname(__file__), '../resources/edge_logo.png')
print(f"Ruta de la imagen: {os.path.abspath(image_path)}")

# Intentar encontrar la imagen en la pantalla
try:
    image_loc = pyautogui.locateCenterOnScreen(image_path, grayscale=False)
    if image_loc:
        print("Se encontró la imagen")
    else:
        print("Imagen no encontrada")
except pyautogui.ImageNotFoundException:
    print("Imagen no encontrada por PyAutoGUI")
except Exception as e:
    print(f"Ocurrió un error: {e}")

"""
