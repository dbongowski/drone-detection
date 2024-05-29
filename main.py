import os
import cv2
import argparse
import numpy as np
from numpy.linalg import inv
import threading
from periphery import SPI
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference


class FreshestFrame(threading.Thread):

    ## zrodlo: https://gist.github.com/crackwitz/15c3910f243a42dcd9d4a40fcdb24e40

    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes        
        self.callback = None
        
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond: # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)


class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):
        self.A = A  # State transition matrix
        self.B = B  # Control matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Observation noise covariance
        self.P = P  # Estimation error covariance
        self.x = x0  # State estimate

    def predict(self, u=None):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        K = np.dot(np.dot(self.P, self.H.T), inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_estimate(self):
        return self.x


class SystemState:
    def _init_(self, last_angle_lr = 0, last_angle_ud = 0, obj_lost = 1, counter = 0):
        self.last_angle_lr = last_angle_lr
        self.last_angle_ud = last_angle_ud
        self.obj_lost = obj_lost
        self.counter = counter

    def set_last_angle_lr(self, last_angle_lr):
        self.last_angle_lr = last_angle_lr

    def get_last_angle_lr(self):
        return self.last_angle_lr

    def set_last_angle_ud(self, last_angle_ud):
        self.last_angle_ud = last_angle_ud

    def get_last_angle_ud(self):
        return self.last_angle_ud

    def set_obj_lost(self, obj_lost):
        self.obj_lost = obj_lost

    def get_obj_lost(self):
        return self.obj_lost

    def set_counter(self, counter):
        self.counter = counter

    def get_counter(self):
        return self.counter

    def inceremet_counter(self, counter):
        self.counter += 1
    

# Zmienne lokalne
DEFAULT_MODEL_DIR = '../all_models'
DEFAULT_MODEL = 'drone_detection.tflite'
DEFAULT_LABELS = 'drone_labels.txt'
CAMERA_IDX = 1
WIDTH = 1280
HEIGHT = 720
THRESHOLD = 0.3
CENTER_COORDINATE = 150
SCALE_LR = 64/15
SCALE_UD = 12/5
COEFFICIENTS_LR = (3.34E-08, -1.25E-05, 0.07601747155, -0.1044659778)
COEFFICIENTS_UD = (7.49E-09, 6.31E-06, 0.07231745113, 0.09520346172)
SCALE_SPI = 546


# Zmienne globalne służące do rejestracji
MEASURE_X_TAB = []
MEASURE_Y_TAB = []
MEASURE_COUNTER = 0


def main():

    #Utwórz i obsłuż parametry wejściowe programu
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_LABELS))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use.', default=CAMERA_IDX)
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help='classifier score threshold')
    args = parser.parse_args()

    print(f'Loading {args.model} with {args.labels} labels.')

    #Utwórz filtr kalmana
    A, H, Q, R, P, x0 = setup_kalman_filter
    kalman_filter = KalmanFilter(A, None, H, Q, R, P, x0)

    #Utwór interpreter sieci neuronowej
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    #Definicja zapisywania klatek
    imnum = 0
    dir = "kalman"
    if not os.path.exists(dir):
        os.makedirs(dir)

    #Utwórz obiekt rejestrujący obraz
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    fresh = FreshestFrame(cap)

    #Zainicjuj stan systemu
    system_state = SystemState()

    #Main loop
    while True:

        _, frame = fresh.read(wait=True)
        if frame is None:
            break

        #Zapisywanie klatek
        imname = "frame" + str(imnum) + ".jpg"
        path = os.path.join(dir, imname)
        imnum += 1
        cv2.imwrite(path, frame)

        #Przetwarzanie obrazu
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

        #Uruchomienie sieci neuronowej
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)

        #Przetwarzanie i wysyłanie pozycji drona
        send_bbox_coordinates(objs, kalman_filter, system_state)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fresh.release()
    cv2.destroyAllWindows()

def setup_kalman_filter():
    dt = 0.15  # 30 fps
    A = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])
    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    Q = np.eye(4)  # Macierz szumów procesu
    Q = Q * 4
    R = np.array([[0.21105229710459336*30, 0], [0, 0.077922530845062*30]])  # Macierz szumów pomiarowych
    P = np.eye(4) * 1000  # Macierz kowariancji błędu estymacji; duża wartość początkowa
    x0 = np.array([0, 0, 0, 0])  # Początkowy stan
    return A, H, Q, R, P, x0


def send_bbox_coordinates(objs, kalman_filter, scale_spi=SCALE_SPI, system_state):
    
    # Inicjalizacja listy
    objs_with_scores = []

    # Dodanie obiektow do listy
    for obj in objs:
        objs_with_scores.append((obj.score, obj))

    # Sortowanie listy na podstawie wyników (score), od najwyższego
    objs_with_scores.sort(key=lambda x: x[0], reverse=True)
    selected_objs = []

    # Wybranie obiektów z najwyższym, drugim i trzecim wynikiem
    selected_objs.append(objs_with_scores[0][1]) if len(objs_with_scores) > 0 else None
    selected_objs.append(objs_with_scores[1][1]) if len(objs_with_scores) > 1 else None
    selected_objs.append(objs_with_scores[2][1]) if len(objs_with_scores) > 2 else None


    if len(selected_objs) > 0:
        
        angle_lr, angle_ud = process_bbox(selected_objs[0])
        kalman_filter.predict()  # Prognoza

        #Warunki dopasowujące bboxy do obiektu, jeżeli pierwszy jest 3 stopnie dalej od ostatniej pozycji
        if (abs(angle_lr - system_state.get_last_angle_lr()) > 3 * scale_spi or abs(angle_ud - system_state.get_last_angle_ud()) > 3 * scale_spi) and system_state.get_obj_lost() == 0 and system_state.get_counter() < 4:
            if len(selected_objs) > 1:

                #Drugi bbox
                angle_lr, angle_ud = process_bbox(selected_objs[1])
                if (abs(angle_lr - system_state.get_last_angle_lr()) < 3 * scale_spi and abs(angle_ud - system_state.get_last_angle_ud()) < 3 * scale_spi):
                    
                    system_state.inceremet_counter()
                    system_state.set_obj_lost(0)
                    kalman_filter.update(np.array([angle_lr/scale_spi, angle_ud/scale_spi]))  # Aktualizacja
                    print("Object 2 is drone")
                elif len(selected_objs) > 2:

                    #Trzeci bbox
                    angle_lr, angle_ud = process_bbox(selected_objs[2])
                    if (abs(angle_lr - system_state.get_last_angle_lr()) < 3 * scale_spi and abs(angle_ud - system_state.get_last_angle_ud()) < 3 * scale_spi):
                        system_state.inceremet_counter()
                        system_state.set_obj_lost(0)
                        kalman_filter.update(np.array([angle_lr/scale_spi, angle_ud/scale_spi]))  # Aktualizacja
                        print("Object 3 is drone")
                    else:
                        system_state.set_obj_lost(1)

        #Pierwszy bbox jest blisko poprzedniej pozycji
        else:           
            system_state.set_counter(0)
            system_state.set_obj_lost(0)
            kalman_filter.update(np.array([angle_lr/scale_spi, angle_ud/scale_spi]))  # Aktualizacja filtru kalmana
        
        #Zapisywanie poprzedniej pozycji    
        system_state.set_last_angle_lr(angle_lr)
        system_state.set_last_angle_ud(angle_ud)
        

        estimate = kalman_filter.get_estimate()  
        # prognoza1 = kalman_filter.predict() # Predykcja 1 w przód
        angle_lr = int16_convert(estimate[0]*scale_spi)
        angle_ud = int16_convert(estimate[2]*scale_spi)

        SPI_transfer(angle_lr, angle_ud)
    
    #Nie wykryto żadnego obiektu
    else:
        system_state.set_obj_lost(2)
        kalman_filter.predict()  # Prognoza
        estimate = kalman_filter.get_estimate()
        angle_lr = int16_convert(estimate[0]*scale_spi)
        angle_ud = int16_convert(estimate[2]*scale_spi)
        print("No detection")

        SPI_transfer(angle_lr, angle_ud)


def process_bbox(obj):

    bbox = obj.bbox
    x0, y0, x1, y1 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

    #Oblicz środek bboxa
    centroid_x = (x0 + x1) / 2
    centroid_y = (y0 + y1) / 2

    #Przetwarzanie danych z pixeli do kąta
    angle_lr = calculate_SPI_angle(centroid_x, SCALE_LR, COEFFICIENTS_LR)
    angle_ud = calculate_SPI_angle(centroid_y, SCALE_UD, COEFFICIENTS_UD)
    return angle_lr, angle_ud


def calculate_SPI_angle(coordinate, scale, coefficients, center=CENTER_COORDINATE, scale_spi=SCALE_SPI):
    a3, a2, a1, a0 = coefficients
    # Oblicz odległość od środka
    temp = abs(coordinate - center)
    # Przeskaluj do rozdzielczości kamery 1280x720
    px = temp * scale
    # Oblicz kąt za pomocą funkcji wielomianowej
    angle = a0 + a1 * px + a2 * px ** 2 + a3 * px ** 3

    # Dodaj odpowiedni znak kąta
    if coordinate < center:
        angle *= -1

    angle *= scale_spi

    int16_angle = int16_convert(angle)
    return int16_angle


def int16_convert(angle):

    #Ograniczenie wartości do zakresu int16
    if angle > np.iinfo(np.int16).max or angle < np.iinfo(np.int16).min:
        angle = np.clip(angle, np.iinfo(np.int16).min, np.iinfo(np.int16).max)

    int16_angle = np.int16(angle)
    return int16_angle


def split_int16(val):

    #Podział wartości na dwa bajty
    if val < 0:
        val += 2**16 
    high_byte = (val & 0xFF00) >> 8
    low_byte = val & 0x00FF
    return high_byte, low_byte


def SPI_transfer(angle_lr, angle_ud):

    # # Zapis i wydruk wysyłanych wartości
    # global MEASURE_X_TAB, MEASURE_Y_TAB, MEASURE_COUNTER\

    # MEASURE_X_TAB.append(angle_lr)
    # MEASURE_Y_TAB.append(angle_ud)
    # MEASURE_COUNTER += 1

    # if MEASURE_COUNTER > 300:
    #     print("measurex="+str(MEASURE_X_TAB))
    #     print("measurey="+str(MEASURE_Y_TAB))
    #     MEASURE_COUNTER = 0

    #Utwórz obiekt SPI
    spi = SPI("/dev/spidev0.0", 0, 500000)

    #Podział wartości na dwa bajty
    high_byte1, low_byte1 = split_int16(angle_lr)
    high_byte2, low_byte2 = split_int16(angle_ud)

    #Utwórz paczkę danych
    data_out = [low_byte1, high_byte1, low_byte2, high_byte2]
    
    # Wyślij dane przez SPI
    data_in = spi.transfer(data_out)
    
    spi.close()


if __name__ == '__main__':
    main()
