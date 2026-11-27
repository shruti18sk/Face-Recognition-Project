import cv2
def read_img(path, width=500):
    """
    Read an image and resize it to a fixed width.

    Parameters:
        path (str): Path to the image.
        width (int): Desired image width.

    Returns:
        numpy.ndarray: Resized image in BGR format.
    """

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Unable to read image: {path}")

    h, w = img.shape[:2]

    if w == 0:
        raise ValueError("Image width cannot be zero.")

    ratio = width / float(w)
    height = int(h * ratio)

    img = cv2.resize(img, (width, height))

    return img

