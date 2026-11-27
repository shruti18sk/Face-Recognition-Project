import os
import cv2
import face_recognition

from src.utils import read_img


def load_reference_images(directory):
    """
    Load reference images and extract face encodings.

    Returns:
        encodings: List of facial encodings.
        names: List of image labels.
    """

    encodings = []
    names = []

    for file in os.listdir(directory):

        path = os.path.join(directory, file)

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = read_img(path)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_encs = face_recognition.face_encodings(rgb_img)

        if len(face_encs) == 0:
            print(f"No face found in: {file}")
            continue

        encodings.append(face_encs[0])
        names.append(os.path.splitext(file)[0])

    return encodings, names


def compare_faces_with_reference(
    reference_encodings,
    test_encoding,
    tolerance=0.6
):
    """
    Compare a test encoding with reference encodings.

    Returns:
        matches: Boolean match results.
        distances: Face distance values.
    """

    matches = face_recognition.compare_faces(
        reference_encodings,
        test_encoding,
        tolerance=tolerance
    )

    distances = face_recognition.face_distance(
        reference_encodings,
        test_encoding
    )

    return matches, distances


def process_image(
    image_path,
    reference_encodings,
    reference_names,
    tolerance=0.6
):
    """
    Detect faces, compare them with reference faces
    and draw bounding boxes with labels.
    """

    img = read_img(image_path)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)

    face_encodings = face_recognition.face_encodings(
        rgb_img,
        face_locations
    )

    results = []

    for location, encoding in zip(
        face_locations,
        face_encodings
    ):

        matches, distances = compare_faces_with_reference(
            reference_encodings,
            encoding,
            tolerance
        )

        label = "Unknown"

        if len(distances) > 0:

            best_index = distances.argmin()

            if matches[best_index]:
                label = reference_names[best_index]

        top, right, bottom, left = location

        cv2.rectangle(
            img,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            label,
            (left, max(top - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        results.append({
            "label": label,
            "distance": float(distances.min())
            if len(distances) > 0 else None
        })

    return img, results

