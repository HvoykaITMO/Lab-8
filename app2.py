import cv2


SQ_SIDE = 200 // 2  # половина стороны квадрата из данных пункта 3 моего варианта


def is_rect_inside(rect_in, rect_out):  # Функция, координатно проверяющая, что одна фигура полностью внутри другой
    x_00, y_00, x_01, y_01 = rect_in
    x_10, y_10, x_11, y_11 = rect_out
    if x_00 >= x_10 and y_00 >= y_10 and x_01 <= x_11 and y_01 <= y_11:
        return True
    return False


cap = cv2.VideoCapture('sample.mp4')
assert cap is not None, "file could not be read, check with os.path.exists()"

tsokotuha = cv2.imread('fly64.png')  # Изображение Цокотухи
fly_w, fly_h = tsokotuha.shape[:2][::-1]
i = 0
while True:
    ret, frame = cap.read()
    if not ret:  # Проверка на конец видео
        print('END')
        break

    '''По нажатии любой клавиши закрываем программу
    (+1 т. к. если ничего не нажимать, то вернётся -1, что по логике питона - True'''
    if cv2.waitKey(50) + 1:
        break

    '''Вычисляем координаты центра нашего кадра для выполнения 3 пункта, а также считаем координаты нашего квадрата'''
    frame_w, frame_h = frame.shape[1], frame.shape[0]
    SQUARE = (frame_w // 2 - SQ_SIDE, frame_h // 2 - SQ_SIDE, frame_w // 2 + SQ_SIDE, frame_h // 2 + SQ_SIDE)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Конвертация в серый для дальнейше обработки

    ''' С помощью серого изображения получаем бинарное изображение, в котором пиксели,
    c интенсивностью >= thresh (110), будут окрашены в maxval (255).
    Так как мы указали тип пороговой обработки THRESH_BINARY_INV, то логика будет инвертирована.
    Возвращается два значения: пороговое значение (thresh) и полученное изображение. Так как нам требуется только
    второе, то берем его, указывая индекс [1].'''
    thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]

    ''' Находим контуры с помощью нашего бинарного изображения, cv2.RETR_EXTERNAL указывает на взятие только "внешних"
    контуров, cv2.CHAIN_APPROX_NONE указывает, что ВСЕ точки контура должны быть сохранены.
    Возвращается два значения: массив точек и вектор иерархии, зависящий от второго параметра (cv2.RETR_EXTERNAL).
    Нам нужен только массив точек, поэтому обращаемся к нему по индексу [0]'''
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    x, y, w, h = (0, 0, 0, 0)  # зададим начальные координаты нашего прямоугольника с центром в фигуре на видео
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)  # Берём максимальный по площади контур
        x, y, w, h = cv2.boundingRect(c)  # По этим точкам получаем параметры прямоугольника с центром в точках контура

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Рисуем прямоугольник
        new_fly_x = x - (fly_w // 2 - w // 2)
        new_fly_y = y - (fly_h//2 - h//2)
        roi = frame[new_fly_y:new_fly_y+fly_h, new_fly_x:new_fly_x+fly_w]
        if roi.shape[:2] == (fly_h, fly_w):
            frame[new_fly_y:new_fly_y+fly_h, new_fly_x:new_fly_x+fly_w] = tsokotuha

        if i % 7 == 0:  # Каждый 7 кадр выводим в консоль координаты центра прямоугольника
            a, b = (x + w) // 2, (y + h) // 2
            print(a, b)
    '''Определяем цвет нашего квадрата, если фигура на видео полностью внутри него,
    то он зелёного цвета, иначе - красного'''
    color = (0, 255, 0) if is_rect_inside(rect_in=(x, y, x+w, y+h), rect_out=SQUARE) else (0, 0, 255)
    cv2.rectangle(frame, SQUARE[:2], SQUARE[2:], color,  2)
    cv2.imshow('ART', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
