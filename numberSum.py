import cv2
import numpy as np
import keras
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012

arrayOfRecognizedDigits = []

def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return (X - x, Y - y)


def unit(v):
    x, y = v
    mag = length(v)
    return (x / mag, y / mag)


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return (x*sc, y*sc)


def add(v, w):
    x, y = v
    X, Y = w
    return (x + X, y + Y)


def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist


class RecognizedDigit:

    def __init__(self, x, y, width, height, passed, frame):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.passed = False
        self.frame = frame

def foundDigit(rd):
    tempX = rd[0]
    tempY = rd[1]
    tempWidth = rd[2]
    tempHeight = rd[3]
    tempRD = None
    for recognizeddigits in arrayOfRecognizedDigits:
        dot1 = [recognizeddigits.x + recognizeddigits.width, recognizeddigits.y + recognizeddigits.height]  # dobijena koordinata za donji desni ugao regiona
        dot2 = [tempX + tempWidth, tempY + tempHeight]  # dobijena koordinata za gornji levi ugao regiona
        distanceBetweenDots = length(vector(dot1, dot2))
        if (distanceBetweenDots < 21):
            return recognizeddigits

    return None


def cnn():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    x_train = x_train / 255
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model


def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255-image


def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255


def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        #print(x, y, w, h)
        if area > 30 and h < 100 and h > 10 and w > 1:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom

            recognized_digit = cv2.boundingRect(contour)

            tempRD = foundDigit(recognized_digit)

            if (tempRD is not None):
                tempRD.x = x
                tempRD.y = y
                tempRD.width = w
                tempRD.height = h
            else:
                rd_temp = RecognizedDigit(x, y, w, h, False, None)
                arrayOfRecognizedDigits.append(rd_temp)

            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 1)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]


    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions


def line_detection(mask):
    edges = cv2.Canny(mask, 75, 150)
    filled_line = cv2.dilate(edges, np.ones((3,3)), iterations=1)
    lines = cv2.HoughLinesP(filled_line, 1, np.pi/180, 75, 150, 5)

    return lines


if __name__ == "__main__":

    sumOfNumbers = 0
    frame_num = 0

    cap = cv2.VideoCapture("video-9.avi")
    cap.set(1, frame_num)  # indeksiranje frejmova
    _, first_frame = cap.read()
    mask = cv2.inRange(first_frame, (160, 0, 0), (255, 100, 100))
    blue_lines = line_detection(mask)
    blue_line1 = blue_lines[0]
    blue_line = blue_line1[0]
    x1 = blue_line[0]
    y1 = blue_line[1]
    x2 = blue_line[2]
    y2 = blue_line[3]
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    #analiza videa frejm po frejm
    print(blue_line)
    print(p1)
    print(p2)
    model = keras.models.load_model("keras_mnist.h5")
    #model = cnn()

    while True:

        frame_num += 1
        ret_val, frame = cap.read()
        cv2.line(frame, (blue_line[0], blue_line[1]), (blue_line[2], blue_line[3]), (0, 0, 255), 2)
        cv2.imshow("linija", frame)
        if ret_val == True:

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_bin = image_bin(grayFrame)
            img, _ = select_roi(frame, img_bin)
            #cv2.imshow("regioni", img)

            for rd in arrayOfRecognizedDigits:
                bottomRight = np.array([rd.x + rd.width, rd.y + rd.height])
                distanceBetweenNumAndLine = pnt2line(bottomRight, p1, p2)
                if (distanceBetweenNumAndLine <= 7):
                    if (rd.passed == False):
                        rd.passed = True
                        # sumOfNumbers
                        picture = img_bin[rd.y-5:bottomRight[1]+5, rd.x-5:bottomRight[0]+5]
                        picture = erode(picture)
                        img = resize_region(picture)
                        scaledImage = scale_to_range(img)
                        vectorImage = matrix_to_vector(scaledImage)
                        predictImage = np.reshape(vectorImage, (1, 784))
                        #predictImage = np.reshape(vectorImage, (1, 28, 28, 1))
                        predictedNumber = model.predict(predictImage)
                        sumOfNumbers += np.argmax(predictedNumber)
            print(sumOfNumbers)
            cv2.waitKey(27)
            cv2.imshow("ajde radi", frame)
        if not ret_val:
            break



    cap.release()

