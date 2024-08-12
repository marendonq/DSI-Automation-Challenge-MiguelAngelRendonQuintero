import pyautogui
import time
import os


from utils.general_functions import resolve_path

time.sleep(3)

def locate_and_click(image_path: str, action: str = 'click', grayscale: bool = False, confidence: float = 0.8) -> bool:
    """
    Localiza una imagen en la pantalla y realiza una acción.
    
    Parámetros:
    - image_path (str): Ruta de la imagen a localizar.
    - action (str): Acción a realizar ('click', 'doubleClick', 'moveTo').
    - grayscale (bool): Si se debe usar la búsqueda en escala de grises.
    - confidence (float): Nivel de confianza para la localización de la imagen.

    Retorna:
    - bool: True si la imagen fue encontrada y se realizó la acción, False en caso contrario.
    """
    image_path = resolve_path(image_path)

    print(f"Ruta de la imagen: {os.path.abspath(image_path)}")

    try:
        image_loc = pyautogui.locateCenterOnScreen(image_path, grayscale=grayscale, confidence=confidence)
        if image_loc:
            if action == 'click':
                pyautogui.click(image_loc)
            elif action == 'doubleClick':
                pyautogui.doubleClick(image_loc)
            elif action == 'rightClick':
                pyautogui.rightClick(image_loc)
            else:
                raise ValueError(f"Acción no soportada: {action}")
            print(f"Se realizó la acción '{action}' en la imagen localizada.")
            return True
        else:
            print("Imagen no encontrada")
            return False
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return False 

def open_program(program_path: str):
    pyautogui.hotkey('win', 'r')
    time.sleep(1)
    pyautogui.typewrite(program_path)
    pyautogui.press('enter')

def open_document(file_path: str, program_path: str):
    open_program(program_path)
    time.sleep(2)
    pyautogui.hotkey('ctrl', 'o')
    time.sleep(1)
    pyautogui.typewrite(resolve_path(file_path))
    pyautogui.press('enter')


# Llamadas a Funciones
image_path = resolve_path('../resources/edge_logo.png')

if locate_and_click(image_path):
    print("La imagen se encontró y se realizó la acción.")
else:
    print("No se encontró la imagen o hubo un error.")






"""
Pendientes:
mañana me voy a encargar de crear la ruta para que pyautogui:

    - abra el explorador de archivos (win + e)
    - acceda a una carpeta en especifico buscandola
    - luego debe de abrir la carpeta, tomar capturas del contenido del archivo (mirar si se puede integrar de una vez con tesser act para realiza un mvp)
    - al finalizar el archivo, debe salir, bajar el mouse al siguiente documento en el explorador de archivos y repetir
    - iniciar desarrollo de la API por medio de FastAPI 
"""
    


