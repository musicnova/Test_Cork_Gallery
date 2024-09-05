import os
import requests
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import matplotlib.colors as mcolors
import os
from scipy.spatial import ConvexHull
# Укажите рабочую директорию с
# обрабатываемыми изображениями
#image_directory = r'/content/drive/MyDrive/Галерея пробки/Интерьеры'
class_names = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'trash', 'fan', 'pier', 'crt', 'plate', 'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag']
class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
class_ids = [id - 1 for id in class_ids]
# Создание словаря ADEK20_dict
ADEK20_dict = dict(zip(class_ids, class_names))
data_classes = [
    # class, id_class, class_type, color
    'floor', 3, 'floor', (255, 0, 0),  # Светлый розовый
    'ceiling', 5, 'ceiling', (0, 255, 0),  # Светлый зеленый
    'walls', 0, 'walls', (0, 0, 255),      # Светлый синий
    'door', 14, 'walls', (139, 69, 19),        # Коричневый
    'bed', 7, 'floor', (255, 228, 196),        # Персиковый
    'window', 8, 'walls', (173, 216, 230),     # Голубой
    'cabinet', 10, 'other', (160, 82, 45),     # Коричневый
    'table', 15, 'other', (222, 184, 135),     # Слоновая кость
    'curtain', 18, 'walls', (255, 182, 193),   # Розовый
    'chair', 19, 'other', (240, 230, 140),     # Светло-желтый
    'painting', 22, 'walls', (255, 218, 185),   # Персиковый
    'sofa', 23, 'other', (255, 160, 122),      # Светло-коралловый
    'shelf', 24, 'walls', (139, 137, 137),     # Темно-серый
    'mirror', 27, 'walls', (192, 192, 192),     # Серебристый
    'rug', 28, 'floor', (255, 99, 71),         # Красный
    'television', 31, 'walls', (100, 150, 200),      # Серый синий
    'books', 33, 'walls', (255, 165, 0),        # Оранжевый
    'desk', 35, 'other', (244, 164, 96),       # Светлый коричневый
    'picture', 42, 'walls', (255, 228, 196),    # Персиковый
    'plant', 17, 'other', (100, 200, 100),     # Зеленый
    'cushion', 39, 'other', (255, 105, 180),   # Ярко-розовый
    'book', 67, 'other', (65, 80, 0),          # Темно-зеленый
    'computer', 74, 'walls', (0, 80, 65),      # Темно-бирюзовый
    'swivel', 75, 'walls', (180, 200, 160),    # Светлый зеленый
    'light', 82, 'other', (10, 50, 110),       # Темно-синий
    'television', 89, 'walls', (165, 180, 30), # Светло-оливковый
    'sculpture', 132, 'other', (180, 135, 150),# Розовато-серый
    'lamp', 36, 'other', (33, 96, 186),        # Темно-синий
    'bookcase', 62, 'walls', (43, 243, 128),   # Ярко-зеленый
    'coffee', 64, 'other', (87, 54, 231),      # Фиолетовый
    'chandelier', 85, 'other', (164, 87, 242), # Светло-фиолетовый
    'ottoman', 97, 'other', (112, 222, 76),    # Ярко-зеленый
    'clock', 148, 'walls', (12, 22, 69),       # Темно-синий
    'vase', 135, 'other', (200, 160, 140),     # Бежевый
    'sconce', 134, 'other', (164, 96, 244),    # Фиолетовый
    'tray',137, 'other', (30, 70, 130),        # ?
    'basket', 112, 'other', (96, 164, 244),    # Светло-голубой
    'plate', 142, 'other', (164, 244, 96),     # Светло-зеленый
    'flower', 66, 'other', (64, 144, 6),       # Ярко-зеленый
    'pot', 125, 'other', (80, 0, 65),          # Темно-бордовый
    'fireplace', 49, 'walls', (2, 22, 222),    # Темно-синий
    'wardrobe', 35, 'walls', (1, 111, 11),     # Темно-зеленый
    'box', 41, 'walls', (76, 143, 109),         # Темно-серый
    'other', 1000, 'other', (0, 0, 0),         # Черный
    'wall_left', 1001, 'walls', (200, 200, 200),      # Серый
    'wall_center', 1002, 'walls', (150, 150, 150),    # Темно-серый
    'wall_right', 1003, 'walls', (100, 100, 100),     # Очень темно-серый
]
# Преобразование списка в DataFrame
df_class = pd.DataFrame(
    [data_classes[i:i + 4] for i in range(0, len(data_classes), 4)],
    columns=['class', 'id_class', 'main_class', 'color']
)


def display_images_in_row(*images):
    """
    Отображает список изображений в ряд с заголовками, соответствующими именам переменных.
    :param images: Изображения (NumPy массивы) передаются как аргументы.
    """
    # Получаем имена переменных из пространства имен
    titles = [name for name in globals() if id(globals()[name]) in [id(image) for image in images]]
    # Настраиваем количество подграфиков
    num_images = len(images)
    plt.figure(figsize=(15, 5))  # Размер окна для отображения
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)  # 1 ряд, num_images колонок
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')  # Отключаем оси
    plt.tight_layout()  # Подгоняем подграфики
    plt.show()
# Функция для предсказания и визуализации
def get_simplified_color_segmentation(seg_map):
    # Берем первый канал
    seg_map = seg_map[:, :, 0]
    segmented_classes=np.unique(seg_map)
    # Получаем список уникальных индексов, которых нет в словаре
    missing_ids = [idx for idx in segmented_classes if idx not in ADEK20_dict]
    # Получаем список распознанных идентификаторов классов, которые отсутствуют в df_class['id_class']
    missing_ids = [id for id in segmented_classes if id not in df_class['id_class'].values]
    # Визуализация результата сегментации
    color_segmentation = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in id_to_color.items():
        color_segmentation[seg_map == label, :] = color
    # Визуализация упрощенной маски
    simplified_color_segmentation=color_segmentation.copy()
    return simplified_color_segmentation

def visualize_contour(mask, simplified_contour):
    # Преобразуем маску в двухоттенковую
    two_shade_mask = np.where(mask > 0, 200, 255).astype(np.uint8)  # 200 для объектов, 255 для фона
    # Преобразуем 2-х цветную маску в RGB
    rgb_image = np.stack((two_shade_mask,) * 3, axis=-1)  # Копируем маску по всем трем каналам
    # Преобразуем RGB изображение в формат, совместимый с OpenCV
    vizualized_contour = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # Наложение упрощенного контура (зеленый цвет)
    cv2.drawContours(vizualized_contour, [simplified_contour], -1, (0, 0, 255), 5)
    return vizualized_contour
def display_images_in_row(*images):
    """
    Отображает список изображений в ряд с заголовками, соответствующими именам переменных.
    :param images: Изображения (NumPy массивы) передаются как аргументы.
    """
    # Получаем имена переменных из пространства имен
    titles = [name for name in globals() if id(globals()[name]) in [id(image) for image in images]]
    # Настраиваем количество подграфиков
    num_images = len(images)
    plt.figure(figsize=(15, 5))  # Размер окна для отображения
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)  # 1 ряд, num_images колонок
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')  # Отключаем оси
    plt.tight_layout()  # Подгоняем подграфики
    plt.show()
# получаем отрезки, ограничивающие чертеж
def get_bbox_features(contour):
    contour = np.array(contour)
    # Создаем список отрезков рамки чертежа и находим на них точки пересечения с нижними ребрами комнаты
    x, y, w, h = cv2.boundingRect(contour)
    x1 = x
    y1 = y
    x2 = x + w - 1
    y2 = y + h - 1
    bbox=x1,y1,x2,y2
    # Определяем углы прямоугольника
    p1 = (x1, y1)  # Верхний левый угол
    p2 = (x2, y1)  # Верхний правый угол
    p3 = (x2, y2)  # Нижний правый угол
    p4 = (x1, y2)  # Нижний левый угол
    bbox_corners=[p1,p2,p3,p4]
    # Создаем сегменты для bounding box
    bbox_segments = [
        (p1, p2),  # Верхняя сторона
        (p2, p3),  # Правая сторона
        (p3, p4),  # Нижняя сторона
        (p4, p1)   # Левая сторона
    ]
    box_diagonal = np.sqrt(w**2 + h**2)
    return bbox, bbox_segments, bbox_corners, box_diagonal

# Функция для проверки, лежит ли точка на границе bounding box
def is_on_bbox(point, bbox):
    x, y, w, h = bbox
    return (point[0] == x or point[0] == x + w or
            point[1] == y or point[1] == y + h)

def remove_redundant_points(contour):
    contour = np.array(contour)  # Преобразуем список в numpy массив
    if len(contour) < 3:
        return contour  # Ничего не делать, если меньше трех точек
    # Список для хранения новых точек
    new_contour = [contour[0]]  # Первая точка всегда добавляется
    for i in range(1, len(contour) - 1):
        p1 = contour[i - 1]
        p2 = contour[i]
        p3 = contour[i + 1]
        # Проверяем, являются ли p1, p2 и p3 коллинеарными
        if not is_collinear(p1, p2, p3):
            new_contour.append(p2)  # Добавляем точку, если она не средняя
    new_contour.append(contour[-1])  # Последняя точка всегда добавляется
    return np.array(new_contour)
def is_collinear(p1, p2, p3):
    # Проверяем коллинеарность с помощью площади треугольника
    return np.isclose((p2[1] - p1[1]) * (p3[0] - p2[0]), (p3[1] - p2[1]) * (p2[0] - p1[0]))
def visualize_points_on_contour(contour, points_on_bbox):
    points_on_bbox=np.array(points_on_bbox)
    # Извлекаем координаты контура
    contour_x, contour_y = contour[:, 0], contour[:, 1]
    # Извлекаем координаты точек на bounding box
    bbox_x, bbox_y = points_on_bbox[:, 0], points_on_bbox[:, 1]
def close_contour(contour):
    # Проверяем, замкнут ли контур
    if not np.array_equal(contour[0], contour[-1]):
        # Если не замкнут, добавляем первую точку в конец
        contour = np.vstack([contour, contour[0]])
    return contour

def convert_to_simple_list(points):
    # Преобразуем в простой список точек
    return [tuple(point.flatten()) for point in points]


def inpaint_color_mask(mask):
    # Преобразование маски в оттенки серого для создания маски инпейнтинга
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Создание бинарной маски, где белые области - это дыры, которые нужно заполнить
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY_INV)
    # Применение инпейнтинга
    inpainted_mask = cv2.inpaint(mask, binary_mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted_mask
def extract_wall_mask(image):
    # Синий канал
    #wall_mask = image.copy()
    wall_mask = (image[:, :, 2] >= 200)
    #wall_mask[wall_mask[:, :, 2] != 100] = 0
    return wall_mask
def create_color_reference(df):
    # Определение размера графика
    num_classes = len(df)
    num_columns = 5  # Количество столбцов
    num_rows = (num_classes + num_columns - 1) // num_columns  # Количество строк
    tile_width = 1 / 3
    tile_height = 1 / 10
    fig, ax = plt.subplots(figsize=(num_columns * tile_width * 3, num_rows * tile_height * 10))
    # Отображение цветов
    for i, row in df.iterrows():
        col = i % num_columns
        row_num = i // num_columns
        color_normalized = tuple(c / 255 for c in row['color'])
        ax.add_patch(plt.Rectangle((col * tile_width, (num_rows - row_num - 1) * tile_height), tile_width, tile_height, color=color_normalized))
        ax.text(col * tile_width + tile_width / 2, (num_rows - row_num - 1) * tile_height + tile_height / 2, f"{row['class']} ({row['id_class']})", va='center', ha='center', fontsize=8, color='white')
    # Настройки осей
    ax.set_xlim(0, num_columns * tile_width)
    ax.set_ylim(0, num_rows * tile_height)
    ax.set_aspect('equal')
    ax.axis('off')
    # Сохранение графика в файл
    plt.savefig('Карта цветов.png' , bbox_inches='tight')
    plt.show()
#create_color_reference(df_class)
# Создание удобных словарей из df_class
id_to_color = dict(zip(df_class['id_class'], df_class['color']))
class_to_id = dict(zip(df_class['class'],df_class['id_class']))
class_to_color = dict(zip(df_class['class'],df_class['color']))
def is_point_on_segment(point, p1, p2):
    """
    Проверяет, лежит ли точка на отрезке, соединяющем точки p1 и p2.
    :param point: Проверяемая точка [x, y]
    :param p1: Первая точка отрезка [x, y]
    :param p2: Вторая точка отрезка [x, y]
    :return: True, если точка лежит на отрезке, иначе False
    """
    # Проверяем, что точка находится между p1 и p2 по обеим координатам x и y
    if min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]):
        # Вычисляем векторные произведения для проверки коллинеарности
        cross_product = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (p2[1] - p1[1])
        if abs(cross_product) < 1e-6:  # допуск для вычислений с плавающей точкой
            return True
    return False
def douglas_peucker(contour, max_vertices):
    epsilon = 0.01  # Начальное значение, будет увеличиваться до достижения max_vertices
    while True:
        approx_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx_contour) <= max_vertices:
            print(epsilon)
            return approx_contour
        epsilon += 0.01
# Функция для проверки, должна ли точка быть вставлена между двумя точками упрощенного контура
def should_insert_between(p1, p2, point):
    within_bbox = (min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and
                    min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]))
    return within_bbox
def douglas_peucker_refine(contour, simplified_contour):
    simplified_contour=close_contour(simplified_contour)
    # Получаем bounding box для исходного контура
    bbox = cv2.boundingRect(contour)
    # Извлекаем все точки на bounding box
    points_on_bbox = [point for point in contour if is_on_bbox(point, bbox)]
    #visualize_points_on_contour(contour, points_on_bbox)
    # Вставляем точки на bounding box в упрощенный контур
    for point in points_on_bbox:
        for i in range(len(simplified_contour) - 1):
            if is_point_on_segment(point, simplified_contour[i], simplified_contour[i + 1]):
                break  # Если точка уже покрыта отрезком, выходим из цикла
        else:  # Если точка не покрыта ни одним сегментом, вставляем
            for i in range(len(simplified_contour) - 1):
                if should_insert_between(simplified_contour[i], simplified_contour[i + 1], point):
                    simplified_contour = np.insert(simplified_contour, i+1, [point], axis=0)
                    break
    simplified_contour = remove_redundant_points(simplified_contour)
    return simplified_contour
def find_and_simplify_contour(mask, max_vertices ):
    # Преобразование маски к типу данных uint8
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour = largest_contour.reshape(-1, 2)
    simplified_contour = douglas_peucker(contour, max_vertices)
    simplified_contour = np.squeeze(simplified_contour)
    # Добавляем точки на bbox, пропущенные Дугласом Пекером
    simplified_contour = douglas_peucker_refine(contour, simplified_contour)
    # Замыкание контура
    if not np.array_equal(simplified_contour[0], simplified_contour[-1]):
        simplified_contour = np.vstack([simplified_contour, simplified_contour[0]])
    return simplified_contour, contour

def get_contour_by_mask (original_image,seg_map):
    # Создадим упрощенную маску сегментации для более эффективного импайнтинга
    #simplified_color_segmentation = get_simplified_color_segmentation(seg_map)
    max_vertices = 11
    # Получаем залитую и упрощенную маску
    inpainted_mask = inpaint_color_mask(seg_map)
    #print(inpainted_mask[200, 200])
    # Выделяем маску стены
    inpainted_wall_mask = extract_wall_mask(inpainted_mask)
    #plt.imshow(inpainted_wall_mask)
    #plt.show()
    # Поиск и упрощение контура
    simplified_contour, contour = find_and_simplify_contour(inpainted_wall_mask,max_vertices)
    return simplified_contour, inpainted_wall_mask, contour
def simplify_contour(contour):
    """
    Упрощает контур, удаляя первую и последнюю точки, если они участвуют
    в горизонтальном отрезке длиной более 5 единиц.
    """
    print(f"Первые три точки контура: {contour[:3]}")
    print(f"Последние три точки контура:{contour[-3:]}")

    if len(contour) < 2:
        return contour  # Возвращаем контур как есть, если в нём меньше двух точек

    # Проверяем первую точку
    if contour[1][1] == contour[0][1] and abs(contour[1][0] - contour[0][0]) > 5:
        print("Удаление первой точки:", contour[0])
        contour = contour[1:]  # Удаляем первую точку, если условие выполнено

    # Проверяем последнюю точку
    if len(contour) > 1 and contour[-1][1] == contour[-2][1] and abs(contour[-1][0] - contour[-2][0]) > 5:
        print("Удаление последней точки:", contour[-1])
        contour = contour[:-1]  # Удаляем последнюю точку, если условие выполнено

    return contour

def adjust_contour(contour):
    """
    Корректирует контур, перемещая точки на основании заданных условий расстояния.

    1. Если расстояние от первой до второй точки контура больше, чем расстояние от первой до последней,
       то переставить первую точку в конец контура.
    2. Если расстояние от последней точки до предпоследней больше, чем от последней до первой,
       то переставляем последнюю точку на первое место контура.
    """
    # Проверяем условия для первой точки
    if np.linalg.norm(contour[0] - contour[1]) > np.linalg.norm(contour[0] - contour[-1]):
        # Перемещаем первую точку в конец
        contour = np.append(contour[1:], [contour[0]], axis=0)

    # Проверяем условия для последней точки после возможной коррекции первой точки
    if np.linalg.norm(contour[-1] - contour[-2]) > np.linalg.norm(contour[-1] - contour[0]):
        # Перемещаем последнюю точку на первое место
        contour = np.append([contour[-1]], contour[:-1], axis=0)

    return contour

def get_floor_ceiling_hulls(contour):
    """
    Обрабатывает контур, разделяет его на пол и потолок, создает выпуклые оболочки,
    и возвращает их как списки точек.
    """
    mean_y = (np.max(contour[:, 1]) + np.min(contour[:, 1])) / 2
    floor_contour = contour[contour[:, 1] > mean_y]
    ceiling_contour = contour[contour[:, 1] < mean_y]

    # Адаптируем потолочный контур если начальная точка совпадает с первой точкой исходного контура
    floor_contour = adjust_contour(floor_contour)
    ceiling_contour = adjust_contour(ceiling_contour)

    # Упрощение контуров
    floor_contour = simplify_contour(floor_contour)
    ceiling_contour = simplify_contour(ceiling_contour)

    floor_hull = None
    ceiling_hull = None

    # Создаем выпуклые оболочки, если это возможно
    if len(floor_contour) > 2:
        floor_hull = ConvexHull(floor_contour)
        floor_hull = floor_contour[floor_hull.vertices]
    if len(ceiling_contour) > 2:
        ceiling_hull = ConvexHull(ceiling_contour)
        ceiling_hull = ceiling_contour[ceiling_hull.vertices]

    return floor_hull, ceiling_hull

    
def conturs_on_image(original_image,seg_map):
    # Упрощаем контур
    simplified_contour, _, contour = get_contour_by_mask(original_image,seg_map)
    # Визуализируем контур
    org_img = original_image.copy()
    floor_contour, ceiling_contour = get_floor_ceiling_hulls(simplified_contour)
    floor_contour_1, ceiling_contour_1 = get_floor_ceiling_hulls(contour)
#    print('ceiling_contour', ceiling_contour)
#    print('ceiling_contour.shape', ceiling_contour.shape)
#    print('type ceiling_contour', type(ceiling_contour))
    #visualized_contour = cv2.drawContours(org_img, [ceiling_contour], -1, (0, 255, 255), 5)
    #visualized_contour = cv2.drawContours(org_img, [floor_contour], -1, (0, 0, 255), 5)
    #visualized_contour = cv2.drawContours(org_img, [simplified_contour], -1, (0, 255, 255), 5)
    visualized_contour = cv2.drawContours(org_img, [floor_contour_1], -1, (255, 0, 255), 5)
    visualized_contour = cv2.drawContours(org_img, [ceiling_contour_1], -1, (255, 255, 0), 5)
    #visualized_contour = cv2.drawContours(org_img, [simplified_contour], -1, (0, 0, 0), 5)
    #visualized_contour = visualize_contour(inpainted_wall_mask, simplified_color)
    # Отображаем результаты
    #display_images_in_row(original_image,visualized_contour)
    return visualized_contour, simplified_contour