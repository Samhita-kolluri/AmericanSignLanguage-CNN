from PIL import Image
from keras.models import load_model
import cv2
import numpy as np
import time

capture = cv2.VideoCapture(0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

model = load_model('Models/old_base_model_asl_50')

result = ""

while capture.isOpened():
    stime = time.time()
    ret, frame_capture = capture.read()
    cv2.imwrite('live_test.jpg', frame_capture)
    frame_file = 'live_test.jpg'
    frame = Image.open(frame_file)
    frame_grey = frame.convert('L')

    input_size = (28, 28)
    frame_csv = frame_grey.resize(input_size)

    value = np.asarray(frame_csv.getdata(), dtype=np.int).reshape((frame_csv.size[1], frame_csv.size[0]))
    value = value.flatten()

    value_norm = value / 255
    input_value = value_norm.reshape(-1, 28, 28, 1)

    if ret:
        output_value = model.predict_classes(input_value)

        if output_value[0] >= 9:
            output_value[0] += 1

        if output_value == 0:
            result = result + "A"

        elif output_value == 1:
            result = result + "B"

        elif output_value == 2:
            result = result + "C"

        elif output_value == 3:
            result = result + "D"

        elif output_value == 4:
            result = result + "E"

        elif output_value == 5:
            result = result + "F"

        elif output_value == 6:
            result = result + "G"

        elif output_value == 7:
            result = result + "H"

        elif output_value == 8:
            result = result + "I"

        elif output_value == 10:
            result = result + "K"

        elif output_value == 11:
            result = result + "L"

        elif output_value == 12:
            result = result + "M"

        elif output_value == 13:
            result = result + "N"

        elif output_value == 14:
            result = result + "O"

        elif output_value == 15:
            result = result + "P"

        elif output_value == 16:
            result = result + "Q"

        elif output_value == 17:
            result = result + "R"

        elif output_value == 18:
            result = result + "S"

        elif output_value == 19:
            result = result + "T"

        elif output_value == 20:
            result = result + "U"

        elif output_value == 21:
            result = result + "V"

        elif output_value == 22:
            result = result + "W"

        elif output_value == 23:
            result = result + "X"

        elif output_value == 24:
            result = result + "Y"

        else:
            result = result + "error"

        if "Error" in result:
            print('Error getting prediction')
            print('Try again with the same alphabet')
            result = result[:-5]

        else:
            print(result)

    else:
        capture.release()
        cv2.destroyAllWindows()
        break

print('Completed run')
