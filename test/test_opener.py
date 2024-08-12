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

 


"""
mañana me voy a encargar de crear la ruta para que pyautogui:

    - abra el explorador de archivos (win + e)
    - acceda a una carpeta en especifico buscandola
    - luego debe de abrir la carpeta, tomar capturas del contenido del archivo (mirar si se puede integrar de una vez con tesser act para realiza un mvp)
    - al finalizar el archivo, debe salir, bajar el mouse al siguiente documento en el explorador de archivos y repetir
    - iniciar desarrollo de la API por medio de FastAPI 
"""
import os
    
from automation.opener import locate_and_click

image_path = os.path.join(os.path.dirname(__file__), '../resources/edge_logo.png') if '__file__' in locals() else os.path.join(os.getcwd(), 'resources/edge_logo.png')

if locate_and_click(image_path, grayscale=True, confidence=0-8):
    print("La imagen se encontró y se realizó la acción.")
else:
    print("No se encontró la imagen o hubo un error.")





