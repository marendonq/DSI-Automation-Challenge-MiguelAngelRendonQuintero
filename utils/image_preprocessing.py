import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from general_functions import resolve_path


def save_image(img, output_path):
    """
    Guarda una imagen en la ruta especificada.

    Args:
        img (numpy.ndarray): Imagen en formato de matriz numpy.
        output_path (str): Ruta completa donde se guardará la imagen.
    
    Returns:
        str: Ruta completa del archivo guardado.
    """
    cv2.imwrite(output_path, img)
    return output_path

def invert_image_colors(img: np.ndarray) -> np.ndarray:
    """
    Inverts the colors of the input image.

    Args:
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The image with inverted colors.
    """
    return cv2.bitwise_not(img)

def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts the input image to grayscale.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarize_image(image: np.ndarray) -> np.ndarray:
    """
    Binarizes the input image using thresholding method.

    Args:
        image (np.ndarray): The input grayscale image as a NumPy array.

    Returns:
        np.ndarray: The binarized image.
    """
    _, binary_image = cv2.threshold(image, 200, 230, cv2.THRESH_BINARY)
    return binary_image

def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Applies a series of morphological operations and median blur to remove noise from an image.

    This function performs dilation followed by erosion, then applies a morphological closing operation, 
    and finally uses a median blur to reduce noise in the image.

    Args:
        image (np.ndarray): The input image as a numpy array.

    Returns:
        np.ndarray: The denoised image.
    """
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 1)

    return image

def dilate_font(image: np.ndarray) -> np.ndarray:
    """
    Applies dilation to the input image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The dilated image.
    """
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image,kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def erode_font(image: np.ndarray) -> np.ndarray:
    """
    Applies erosion to the input image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The eroded image.
    """
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image,kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def getSkewAngle(cvImage: np.ndarray) -> float:
    """
    Calculate the skew angle of a given image.

    This function preprocesses the input image by applying Gaussian blur, thresholding, and dilation. 
    It then finds contours and determines the angle of the largest contour's
    bounding box to estimate the skew angle of the image.

    Args:
        cvImage (np.ndarray): The input image as a NumPy array (OpenCV image).

    Returns:
        float: The skew angle of the image in degrees. Positive values indicate a counterclockwise skew, 
               and negative values indicate a clockwise skew.
    """
    # https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a given image by a specified angle.

    This function rotates the input image around its center by the given angle using an affine transformation.

    Args:
        cvImage (np.ndarray): The input image as a NumPy array (OpenCV image).
        angle (float): The angle in degrees by which to rotate the image. Positive values rotate counterclockwise.

    Returns:
        np.ndarray: The rotated image as a NumPy array.
    """
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew_image(cvImage: np.ndarray) -> np.ndarray:
    """
    Deskew a given image by correcting its skew angle.

    This function calculates the skew angle of the input image, then rotates the image by the negative of
    that angle to straighten it.

    Args:
        cvImage (np.ndarray): The input image as a NumPy array (OpenCV image).

    Returns:
        np.ndarray: The deskewed image as a NumPy array.
    """
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def remove_borders(image: np.ndarray) -> np.ndarray:
    """
    Removes borders from the input image by cropping to the largest contour.
    If no significant borders are detected, the original image is returned.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The cropped image without borders, or the original image if no borders are found.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # No contours found, return the original image
    
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Check if the contour already covers the entire image
    if w == image.shape[1] and h == image.shape[0]:
        return image  # The largest contour already covers the whole image
    
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def add_missing_borders(image: np.ndarray) -> np.ndarray:
    """
    Adds missing borders to the input image. If borders are already present,
    the original image is returned without modification.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The image with added borders, or the original image if borders are already present.
    """
    h, w = image.shape[:2]
    border_size = int(0.05 * min(h, w))  # Adjust the border size based on the image size
    
    # Check if the corners already have the border color (white)
    if (image[0, 0] == 255).all() and (image[0, -1] == 255).all() and (image[-1, 0] == 255).all() and (image[-1, -1] == 255).all():
        return image  # Borders are already present, return the original image
    
    bordered_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered_image

def calculate_average_thickness(image: np.ndarray) -> float:
    """
    Calculates the average thickness of the text in the image.

    Args:
        image (np.ndarray): The input binary image with text.

    Returns:
        float: The average thickness of the text lines.
    """
    # Invert image if needed
    inverted_image = cv2.bitwise_not(image)
    
    # Detect contours of the text
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the thickness of each contour
    thicknesses = []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        thicknesses.append(min(w, h))
    
    # Return the average thickness
    if thicknesses:
        return np.mean(thicknesses)
    
    else:
        return 0.0

def should_apply_thin_font(image: np.ndarray, threshold: float = 1200) -> bool:
    """
    Determines if the text in the image needs thinning.

    Args:
        image (np.ndarray): The input binary image with text.
        threshold (float): The thickness threshold below which thinning is applied.

    Returns:
        bool: True if thinning should be applied, False otherwise.
    """
    average_thickness = calculate_average_thickness(image)
    return average_thickness > threshold

def should_apply_thick_font(image: np.ndarray, threshold: float = 1200) -> bool:
    """
    Determines if the text in the image needs thickening.

    Args:
        image (np.ndarray): The input binary image with text.
        threshold (float): The thickness threshold above which thickening is applied.

    Returns:
        bool: True if thickening should be applied, False otherwise.
    """
    average_thickness = calculate_average_thickness(image)
    return average_thickness < threshold

def preprocess_image(image_path):
    """
    Performs the complete preprocessing of an image for OCR using modular functions, including
    text thickness adjustment based on calculated average thickness.

    Args:
        image_path (str): Path to the original image file.

    Returns:
        numpy.ndarray: Preprocessed image ready for OCR.
    """
    img = cv2.imread(image_path)

    # Apply the modular preprocessing functions
    gray_image = grayscale(img)
    
    binary_image = binarize_image(gray_image)
    
    # Check if text needs thickening or thinning
    if should_apply_thin_font(binary_image):
        # Apply erosion if the text is too thick
        processed_image = erode_font(binary_image)
    elif should_apply_thick_font(binary_image):
        # Apply dilation if the text is too thin
        processed_image = dilate_font(binary_image)
    else:
        processed_image = binary_image

    denoised_image = remove_noise(processed_image)
    borderless_image = remove_borders(denoised_image)
    deskewed_image = deskew_image(borderless_image)

    final_image = add_missing_borders(deskewed_image)

    return final_image

def display(im_path: str) -> np.ndarray:
    """
    Muestra una imagen en una figura de Matplotlib, ajustando el tamaño de la figura a las dimensiones de la imagen.

    Parámetros:
    -----------
    im_path : str
        La ruta al archivo de imagen que se desea mostrar.

    Retorno:
    --------
    fig : matplotlib.figure.Figure
        La figura de Matplotlib que contiene la imagen mostrada.
    """
    # Resolver la ruta de la imagen
    im_path = resolve_path(im_path)

    if not os.path.exists(im_path):
        raise FileNotFoundError(f"El archivo de imagen no fue encontrado en la ruta especificada: {im_path}")

    dpi = 80
    im_data = plt.imread(im_path)
    height, width = im_data.shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()

    return fig


# Example of using the function
image_path = resolve_path('../DSI_AUTO/resources/page_01_ladeada.jpg')
processed_image = preprocess_image(image_path)

# Save the final image if needed
output_path = resolve_path("../DSI_AUTO/resources/final_processed_image.png")
cv2.imwrite(output_path, processed_image)

# Display the final processed image
#display(output_path)"""