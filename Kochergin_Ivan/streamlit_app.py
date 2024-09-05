import streamlit as st
import moduls.Texture as texture
from PIL import Image
import mmcv
import moduls.model
from moduls.texture_for_point import texture_transformation_1
from main import model
import numpy as np 
import cv2
import os
from streamlit_image_select import image_select
from streamlit_image_annotation import pointdet
from moduls.conturs import conturs_on_image
from streamlit_image_comparison import image_comparison
from moduls.settings import path_for_streamlit
st.set_page_config(layout="wide")


@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    img = mmcv.imread(np.array(img))
    return img 
def save_file_st(path_file):
    file_name = path_file.name
    img = load_image(path_file)
    im = Image.fromarray(img)
    output_filename = 'processing.jpg' 
    im.save(output_filename)
    return output_filename
@st.dialog("Выбор текстуры пола", width =  'large')
def select_floor(img_list_1, image_1, img_list):
    tex_floor = image_select("Выбор текстуры пола", img_list_1, return_value= 'index')
    if st.button("Выбрать"):
        st.session_state.json_list[0] = img_list[tex_floor]
        modificate_image('floor')
        st.session_state.select_floor = [tex_floor]
        st.rerun()
@st.dialog("Выбор текстуры стен", width =  'large')
def select_wall(img_list_1, image_1, img_list):
    tex_wall = image_select("Выбор текстуры стен", img_list_1, return_value= 'index')
    if st.button("Выбрать"):
        st.session_state.json_list[1] = img_list[tex_wall]
        modificate_image('wall')
        st.session_state.select_wall = [tex_wall]
        st.rerun()
@st.dialog("Выбор текстуры потолка", width =  'large')
def select_ceiling(img_list_1, image_1, img_list):
    tex_ceiling = image_select("Выбор текстуры потолка", img_list_1, return_value= 'index')
    if st.button("Выбрать"):
        st.session_state.json_list[2] = img_list[tex_ceiling]
        modificate_image('ceiling')
        st.session_state.select_ceiling = [tex_ceiling]
        st.rerun()
def points_sorter(points):
    messages_list = ['Недостаточно точек. Добавьте точек!',
                    'Точек не четное количество. Добавьте или уберите точку!',
                    'Точек слишком много. Уберите лишние точки!',
                    'Точки успешно отсортированы!',
                    'Пол или потолок имеют обратную перспективу. Скорректируйте точки!']
    print('len(points)', len(points))
    print(points)
    if (num_points := len(points)) < 6:
        return [], []
    elif num_points%2 != 0:
        return [], []
    elif num_points > 8:
        return [], []
    else:
        # Сортируем список по y, а при равных y по х
        sorted_list = sorted(points, key=lambda k: [k[1], k[0]])
        print('sorted_list', sorted_list)
        # Делим список пополам и сортируем по x
        ceiling_points = sorted(sorted_list[:num_points//2], key=lambda k: [k[0], k[1]])
        floor_points = sorted(sorted_list[num_points//2:], key=lambda k: [k[0], k[1]])
        bad_points = []
        print('ceiling_points, floor_points', ceiling_points, floor_points)
        # Проверяем направление перспективы потолка
        if ceiling_points[0][1] > ceiling_points[1][1]:
            bad_points.append(ceiling_points[0])
            bad_points.append(ceiling_points[1])
        if ceiling_points[-1][1] > ceiling_points[-2][1]:
            bad_points.append(ceiling_points[-1])
            bad_points.append(ceiling_points[-2])
        # Проверем направление перспективы пола
        if floor_points[0][1] < floor_points[1][1]:
            bad_points.append(floor_points[0])
            bad_points.append(floor_points[1])
        if floor_points[-1][1] < floor_points[-2][1]:
            bad_points.append(floor_points[-1])
            bad_points.append(floor_points[-2])
        c_f_point = [ceiling_points, floor_points]
    return c_f_point
    #if len(bad_points) != 0:
    #  return ceiling_points, floor_points
    #else:
    #  return ceiling_points, floor_points
#@st.dialog('Определение углов комнаты', width='large')
def corner_change_2(image_path):
    list_1 = ['Угол']
    list_label_1 = []
    list_label_2 = []
    list_point_1 = []
    list_point_2 = []
    if 'ceiling_point' in st.session_state and 'floor_point' in st.session_state and st.session_state.ceiling_point is not None and st.session_state.floor_point is not None:
        list_point_1 = st.session_state.ceiling_point
        print('list_point_1', list_point_1)
        for i in range(len(st.session_state.ceiling_point)):
            list_label_1.append(0)
        list_point_2 = st.session_state.floor_point
        for i in range(len(st.session_state.floor_point)):
            list_label_2.append(0)
    else:
            list_point_1 = []
            list_point_2 = []
    list_point = list_point_1 + list_point_2
    list_label = list_label_1 + list_label_2
    print('list_point', list_point)
    print('list_label', list_label)
    point = pointdet(image_path , list_1, 
                                    points=list_point,
                                    #points = [],
                                    labels=list_label,
                                    #labels = [],
                                    use_space=True)
    if point is not None:
        st.session_state.point = point
        st.session_state.floor_point = []
        st.session_state.ceiling_point = []
    #if 'points' not in st.session_state:
        st.session_state.points = []
    if point is not None:
        for i in st.session_state.point:
            if i['label_id'] == 0:
                #st.session_state.ceiling_point.append([int(i['point'][0]), int(i['point'][1])])
                st.session_state.points.append([int(i['point'][0]), int(i['point'][1])])
            else:
                #st.session_state.floor_point.append([int(i['point'][0]), int(i['point'][1])])
                st.session_state.points.append([int(i['point'][0]), int(i['point'][1])])
        print('st.session_state.points', st.session_state.points)
        c_f_point = points_sorter(st.session_state.points)
        st.session_state.ceiling_point = c_f_point[0]
        st.session_state.floor_point = c_f_point[1]
        #print('a', a)
        #st.session_state.ceiling_point = sort_list_ceiling(st.session_state.ceiling_point)
        print(st.session_state.ceiling_point)
        #st.session_state.floor_point = sort_list_floor(st.session_state.floor_point)
        #st.session_state.ceiling_point = cv2.convexHull(np.array(st.session_state.ceiling_point))
        #st.session_state.floor_point = cv2.convexHull(np.array(st.session_state.floor_point))
        #print('st.session_state.ceiling_point', st.session_state.ceiling_point)
        #print('st.session_state.floor_point', st.session_state.floor_point)
        #st.rerun()
def sort_list_ceiling(no_change_list):
    if len(no_change_list) >= 4:
        no_change_list.sort()
        if no_change_list[0][1] > no_change_list[1][1]:
            no_change_list[0], no_change_list[1] = no_change_list[1], no_change_list[0]
        if no_change_list[-1][1] > no_change_list[-2][1]:
            no_change_list[-1], no_change_list[-2] = no_change_list[-2], no_change_list[-1]
    if len(no_change_list) == 3:
        no_change_list.sort()
        if no_change_list[0][1] > no_change_list[1][1]:
            no_change_list[0], no_change_list[1] = no_change_list[1], no_change_list[0]
    return no_change_list
def sort_list_floor(no_change_list):
    if len(no_change_list) >= 4:
        no_change_list.sort()
        if no_change_list[0][1] < no_change_list[1][1]:
            no_change_list[0], no_change_list[1] = no_change_list[1], no_change_list[0]
        if no_change_list[-1][1] < no_change_list[-2][1]:
            no_change_list[-1], no_change_list[-2] = no_change_list[-2], no_change_list[-1]
    if len(no_change_list) == 3:
        no_change_list.sort()
        if no_change_list[0][1] < no_change_list[1][1]:
            no_change_list[0], no_change_list[1] = no_change_list[1], no_change_list[0]
    return no_change_list
#@st.dialog('Определение углов комнаты', width='large')
def corner_change(image_path):
    list_1 = ['Угол потолка', 'Угол пола']
    list_label_1 = []
    list_label_2 = []
    list_point_1 = []
    list_point_2 = []
    if 'ceiling_point' in st.session_state and 'floor_point' in st.session_state and st.session_state.ceiling_point is not None and st.session_state.floor_point is not None:
        list_point_1 = st.session_state.ceiling_point
        for i in range(len(st.session_state.ceiling_point)):
            list_label_1.append(0)
        list_point_2 = st.session_state.floor_point
        for i in range(len(st.session_state.floor_point)):
            list_label_2.append(1)
    else:
            list_point_1 = []
            list_point_2 = []
    list_point = list_point_1 + list_point_2
    list_label = list_label_1 + list_label_2
    print('list_point', list_point)
    print('list_label', list_label)
    point = pointdet(image_path , list_1, 
                                    points=list_point,
                                    #points = [],
                                    labels=list_label,
                                    #labels = [],
                                    use_space=True)
    if point is not None:
        st.session_state.point = point
        st.session_state.floor_point = []
        st.session_state.ceiling_point = []
        
    if point is not None:
        for i in st.session_state.point:
            if i['label_id'] == 0:

                st.session_state.ceiling_point.append([int(i['point'][0]), int(i['point'][1])])
            else:
                st.session_state.floor_point.append([int(i['point'][0]), int(i['point'][1])])
        st.session_state.ceiling_point = sort_list_ceiling(st.session_state.ceiling_point)
        print(st.session_state.ceiling_point)
        st.session_state.floor_point = sort_list_floor(st.session_state.floor_point)
        #st.session_state.ceiling_point = cv2.convexHull(np.array(st.session_state.ceiling_point))
        #st.session_state.floor_point = cv2.convexHull(np.array(st.session_state.floor_point))
        print('st.session_state.ceiling_point', st.session_state.ceiling_point)
        print('st.session_state.floor_point', st.session_state.floor_point)
        #st.rerun()
    #return point
@st.dialog("modification_photo")
def modification_photo(image_1):
    st.session_state.modification_photo = [image_1]
    st.rerun()
def a(path_photo):
    st.session_state.a = path_photo
def predict(path_photo, model):
    path_file =save_file_st(path_photo)
    _, seg_map, seg_all, image = moduls.model.predict_and_visualize(model, path_file)
    print(seg_map.shape)
    st.session_state.predict =[seg_all, image]
    contur_image, simplified_contour = conturs_on_image(image, seg_map)
    st.session_state.contur_image =contur_image
    st.session_state.simplified_contour = simplified_contour
    return image
def modificate_image(surface):
    original_img = cv2.cvtColor(st.session_state.predict[1], cv2.COLOR_BGR2RGB)
    if st.session_state.f_h !=0:
        f_h = st.session_state.f_h
    else:
        f_h = None
    if st.session_state.w_h !=0:
        w_h = st.session_state.w_h
    else:
        w_h = None
    if st.session_state.c_h !=0:
        c_h = st.session_state.c_h
    else:
        c_h = None
    image_1 = texture_transformation_1(original_img, st.session_state.predict[0],
                                    st.session_state.room_l*100, st.session_state.room_w*100, st.session_state.room_h*100,
                                    st.session_state.ceiling_point, st.session_state.floor_point,
                                    st.session_state.floor_angel, st.session_state.wall_angel, st.session_state.ceiling_angel,
                                    st.session_state.json_list, surface = st.session_state.surface,
                                    h_f= f_h, h_w = w_h, h_c = c_h,
                                    rescale = 1.)
    st.session_state.image_1 = image_1
    st.session_state.modification_photo = [image_1]
def save_angel():
    st.session_state.floor_angel = floor_angel
    st.session_state.ceiling_angel = ceiling_angel
    st.session_state.wall_angel = wall_angel
path_icon, path_json, path_logo = path_for_streamlit()
if 'json_list' not in st.session_state:
    st.session_state.json_list = [None, None, None]
img_list_1 = []
img_list = []
for path_file in os.listdir(path_icon):
    img_list_1.append(path_icon + f'\{path_file}')
for path_json_file in os.listdir(path_json):
    if path_json_file.endswith(('.json')):
        img_list.append(path_json + f'\{path_json_file}')
img_list_1.sort()
img_list.sort()
title_col1, title_col2 = st.columns([1,3])
with title_col1:
    st.image(path_logo)
with title_col2:
    st.title(":orange[Замена текстур на фотографии вашей комнаты]",)

change_col_1, change_col_2, change_col_3 = st.columns([1, 4, 1])
# Загрузка фотографиии
with change_col_1:
    path_photo = st.file_uploader("Загрузка фото", type=['png','jpeg','jpg'], accept_multiple_files=False, key=None,
                  help=None, on_change=None, args=None, kwargs=None,
                  disabled=False, label_visibility = 'collapsed')
    if path_photo == None:
        st.session_state.a = None
    if path_photo is not None:
        if st.session_state.a != path_photo:
            st.session_state.ceiling_point = None
            st.session_state.floor_point = None
            st.session_state.a = path_photo
            predict(st.session_state.a, model)
            st.session_state.modification_photo = [cv2.cvtColor(st.session_state.predict[1], cv2.COLOR_BGR2RGB)]
# Выбор угловых точек на фотографии
with change_col_2:
    if path_photo is not None:
            cv2.imwrite('corners.png',st.session_state.contur_image)
            st.session_state.corners = corner_change('corners.png')
            #st.session_state.corners = corner_change_2('corners.png')
# Размеры комнаты и выбор способа набора текстуры
with change_col_3:
    if path_photo is not None:
        st.session_state.room_l = st.number_input('Длина комнаты', min_value=0., max_value=6., value=4., step=0.1, format=None, key='room_l_key', help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
        st.session_state.room_w = st.number_input('Ширина комнаты', min_value=0., max_value=6., value=3., step=0.1, format=None, key='room_w_key', help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
        st.session_state.room_h = st.number_input('Высота потолка', min_value=0., max_value=4., value=3., step=0.1, format=None, key='room_h_key', help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
        # Выбор текстур
col1, col2, col3 = st.columns(3)
# Текстура пола
with col1:
    if path_photo is not None:
        floor = st.checkbox('Текстура пола', value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if floor:
            floor_select = st.button("Выбор текстуры пола", key = '1')
            if floor_select:
                    select_floor(img_list_1, cv2.cvtColor(st.session_state.modification_photo[0], cv2.COLOR_BGR2RGB), img_list)
            try:
                st.image(img_list_1[st.session_state.select_floor[0]], caption=None, width=200, use_column_width=None, clamp=False, channels='RGB', output_format="auto")
                st.session_state.json_list[0] = img_list[st.session_state.select_floor[0]]
            except:
                b=1
        else:
            st.session_state.json_list[0] = None
# Текстура стен
with col2:
    if path_photo is not None:
        wall = st.checkbox('Текстура стен', value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if wall:
            wall_select = st.button("Выбор текстуры стен", key = '2')
            if wall_select:
                select_wall(img_list_1, cv2.cvtColor(st.session_state.modification_photo[0], cv2.COLOR_BGR2RGB), img_list)
            try:
                st.image(img_list_1[st.session_state.select_wall[0]], caption=None, width=200, use_column_width=None, clamp=False, channels='RGB', output_format="auto")
                st.session_state.json_list[1] = img_list[st.session_state.select_wall[0]]
            except:
                b=1
        else:
            st.session_state.json_list[1] = None
# Текстура потолка
with col3:
    if path_photo is not None:
        ceiling = st.checkbox('Текстура потолка', value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if ceiling:
            ceiling_select = st.button("Выбор текстуры потолка", key = '3')
            if ceiling_select:
                    select_ceiling(img_list_1, cv2.cvtColor(st.session_state.modification_photo[0], cv2.COLOR_BGR2RGB), img_list)
            try:
                st.image(img_list_1[st.session_state.select_ceiling[0]], caption=None, width=200, use_column_width=None, clamp=False, channels='RGB', output_format="auto")
                st.session_state.json_list[2] = img_list[st.session_state.select_ceiling[0]]
            except:
                b=1
        else:
            st.session_state.json_list[2] = None
if path_photo is not None:
    st.session_state.surface = [floor, wall, ceiling]

# Выбор углов поворота текстур
col1_1, col2_1 = st.columns([3,1])
try:
    with col2_1:
        if path_photo is not None:
            floor_angel = st.slider(label = 'Угол поворота текстуры пола',min_value=0, max_value=360, value=0, step=1, format=None, key='floor_angle_key',
                    help=None, on_change=save_angel, args=None, kwargs=None, disabled=False, label_visibility= 'visible')
            wall_angel = st.slider(label = 'Угол поворота текстуры стен',min_value=0, max_value=360, value=0, step=1, format=None, key='wall_angle_key',
                    help=None, on_change=save_angel, args=None, kwargs=None, disabled=False, label_visibility= 'visible')
            ceiling_angel = st.slider(label = 'Угол поворота текстуры потолка',min_value=0, max_value=360, value=0, step=1, format=None, key='ceiling_angle_key',
                    help=None, on_change=save_angel, args=None, kwargs=None, disabled=False, label_visibility= 'visible')
            st.write(":orange[Выбор способа набора текстур]")
            st.write(":orange[Если больше 1 то набор ёлочкой]")
            st.session_state.f_h = st.number_input('Пол', min_value=0, max_value=3, value=0, step=1, format=None, key='f_h_key', help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
            st.session_state.w_h = st.number_input('Стены', min_value=0, max_value=3, value=0, step=1, format=None, key='w_h_key', help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
            st.session_state.c_h = st.number_input('Потолок', min_value=0, max_value=3, value=0, step=1, format=None, key='c_h_key', help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

            if 'floor_angel' not in st.session_state:
                st.session_state.floor_angel = 0
            if 'wall_angel' not in st.session_state:
                st.session_state.wall_angel = 0
            if 'ceiling_angel' not in st.session_state:
                st.session_state.ceiling_angel = 0
            try:
                if st.session_state.floor_angel != floor_angel:
                    modificate_image('floor')
            except:
                print('no_change_angel_floor')
            try:
                if st.session_state.wall_angel != wall_angel:
                    modificate_image('floor')
            except:
                print('no_change_angel_wall')
            try:
                if st.session_state.ceiling_angel != ceiling_angel:
                    modificate_image('ceiling')
            except:
                print('no_change_angel_ceiling')

# Отображение фотографии с наложением текстур    
    with col1_1:
    #    scroll = st.slider(label = '123',min_value=0., max_value=1., value=0.5, step=0.01, format=None, key=33, 
    #            help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility= 'hidden')
        if path_photo is not None:
            if st.session_state.modification_photo is not None:
                image_1 = st.session_state.modification_photo[0]
            else:
                image_1 = cv2.cvtColor(st.session_state.predict[1], cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(st.session_state.predict[1], cv2.COLOR_BGR2RGB)
            #st.write(st.get_option())
            image_comparison(
                            img1=image,
                            img2=image_1,
                            width = 1080,
                            starting_position = 1
                            )
            #fix_line = np.zeros((image.shape[0], 5, 3), dtype=np.uint8)
            #fix_line[:,:,1]  = 254
            #image_2 =cv2.hconcat([image[:, 0:int(image.shape[1]*scroll)], fix_line])
            #image_3 =cv2.hconcat([image_2, image_1[: , int(image_1.shape[1]*scroll+5):]])
            #img = st.image(image_3, caption=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
            cv2.imwrite('Change_photo.jpg', cv2.cvtColor(st.session_state.modification_photo[0], cv2.COLOR_BGR2RGB))
            with open('Change_photo.jpg', "rb") as file:
                print(type(file))
                btn = st.download_button(
                        label="Скачать изображение с наложенной текстурой",
                        data=file,
                        file_name="Change_photo.png",
                        mime="image/png"
                    )
except:
    b=1

