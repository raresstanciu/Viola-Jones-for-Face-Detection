# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import skimage as skl
import sklearn as sklrn
import cv2 as cv



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def open_image(path):
    image = cv.imread(path)
    return image

def main():
    original_image = open_image('./faces.jpg')

    # visualize image
    cv.imshow('Image', open_image('./faces.jpg'))

    # convert the image to grayscale for Viola-Jones
    gray_image = cv.cvtColor(open_image('./faces.jpg'), cv.COLOR_BGR2GRAY)

    # show the converted image
    cv.imshow('Gray Image', gray_image)

    # load the Viola-Jones classifier for face detection
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # detect faces in the image
    detected_faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    print(f'I am able to detect {len(detected_faces)} faces')

    # show the faces in the image
    for (x, y, w, h) in detected_faces:
        cv.rectangle(original_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv.imshow('Detected Faces', original_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
