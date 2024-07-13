import cv2
import numpy as np
def fill_texture_in_perspective(path_to_texture,original_image):
  # Загрузка текстурного изображения
  texture = cv2.imread(path_to_texture)
  #print(texture.shape)
  tex = np.zeros((texture.shape[0]*2, texture.shape[1]*2, 3),dtype=np.uint8)
  tex[0:texture.shape[0], 0:texture.shape[1]] = texture
  tex[texture.shape[0]:2*texture.shape[0], 0:texture.shape[1]] = texture
  tex[0:texture.shape[0], texture.shape[1]:2*texture.shape[1]] = texture
  tex[0:texture.shape[0], 0:texture.shape[1]] = texture
  texture = tex
  if texture is None:
      print("Не удалось загрузить текстурное изображение.")
      exit()
  width, height,_ = original_image.shape
  # Размеры текстуры и полотна
  texture_size = 600
  plane_width = max(width, height)*2
  if plane_width*0.35<height:
    plane_width=int(height/0.35)
  plane_height = plane_width

  # Создание пустого полотна
  plane = np.zeros((plane_height, plane_width, 3), dtype=np.uint8)

  # Наложение текстуры на полотно с учетом граничных условий
  for y in range(0, plane_height, texture_size):
      for x in range(0, plane_width, texture_size):
          # Вычисляем размер текстурного кусочка
          texture_patch_width = min(texture_size, plane_width - x)
          texture_patch_height = min(texture_size, plane_height - y)

          # Вырезаем кусочек текстуры
          texture_patch = texture[:texture_patch_height, :texture_patch_width]

          # Вставляем текстуру на полотно
          plane[y:y+texture_patch_height, x:x+texture_patch_width] = texture_patch

  # Координаты исходных точек
  src_points = np.float32([
      [0, 0],
      [plane_width, 0],
      [plane_width, plane_height],
      [0, plane_height]
  ])

  # Координаты целевых точек
  dst_points = np.float32([
      [plane_width * 0.25, plane_height * 0.45],  # Верхний левый угол смещен вправо
      [plane_width * 0.75, plane_height * 0.45],  # Верхний правый угол смещен влево
      [plane_width, plane_height],    # Нижний правый угол без изменений
      [0, plane_height]         # Нижний левый угол без изменений
  ])

  # Вычисление матрицы перспективного преобразования
  M = cv2.getPerspectiveTransform(src_points, dst_points)

  # Применение перспективного преобразования
  warped_image = cv2.warpPerspective(plane, M, (plane_width, plane_height))

  # Координаты для вырезания прямоугольника
  x, y = int(plane_width * 0.25), int(plane_height * 0.65)
  width, height = int(plane_width * 0.75 - plane_width * 0.25), int(plane_height - plane_height * 0.65)

  # Вырезание прямоугольника из изображения
  texture_image = cv2.getRectSubPix(warped_image, (width, height), (x + width/2, y + height/2))

  return cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB),cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

def transform_texture(original_image,texture_image, floor_mask):
  # print('original_image.shape,floor_mask.shape, texture_image.shape',original_image.shape,floor_mask.shape, texture_image.shape)

  original_image = np.array(original_image)
  
  # Пропорциональное масштабирование текстуры под исходное изображение
  scale_factor = max(original_image.shape[0] / texture_image.shape[0], original_image.shape[1] / texture_image.shape[1])

  rescaled_texture = cv2.resize(texture_image, (int(texture_image.shape[1] * scale_factor), int(texture_image.shape[0] * scale_factor)))
  #print(f'rescaled_texture.shape = {rescaled_texture.shape}')
  # Нахождение центров масштабированной текстуры и исходного изображения
  center_original = (original_image.shape[1] // 2, original_image.shape[0] // 2)
  center_texture = (rescaled_texture.shape[1] // 2, rescaled_texture.shape[0] // 2)

  # Определение области обрезки для текстуры
  start_x = max(center_texture[0] - center_original[0], 0)
  start_y = max(center_texture[1] - center_original[1], 0)
  end_x = min(center_texture[0] + original_image.shape[1] - center_original[0], rescaled_texture.shape[1])
  end_y = min(center_texture[1] + original_image.shape[0] - center_original[1], rescaled_texture.shape[0])

  # Обрезка текстуры
  texture_image = rescaled_texture[start_y:end_y, start_x:end_x]
  #print(f'texture_image.shape = {texture_image.shape}')
  # Преобразование цветных пикселей пола в серую гамму по маске пола
  gray_floor = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
  gray_floor_colored = cv2.cvtColor(gray_floor, cv2.COLOR_GRAY2BGR)  # Преобразование обратно в трехканальное изображение

  # Создание копии оригинального изображения для смешивания
  blended_image = original_image.copy()
  blended_image[floor_mask > 0] = texture_image[floor_mask > 0]

  # Задание пропорции смешивания
  alpha = 0.2  # Пример пропорции (50% оригинальное изображение, 50% текстура)

  # Смешивание серого пола с текстурой по маске пола
  #blended_image[floor_mask > 0] = cv2.addWeighted(gray_floor_colored[floor_mask > 0], alpha, texture_image[floor_mask > 0], 1 - alpha, 0)

  return blended_image
def texture_perspective(image, path_to_texture, mask):
    _,texture_image= fill_texture_in_perspective(path_to_texture,image)
    blended_image  = transform_texture(image,texture_image, mask)
    #blended_image_path =cv2.imwrite(r'C:\Project\Corn\photo_out\1.jpg', blended_image)
    return blended_image