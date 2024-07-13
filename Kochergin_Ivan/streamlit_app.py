import streamlit as st
import moduls.Texture as texture
from PIL import Image
import mmcv
import moduls.model
from main import model
import numpy as np 
import cv2
import time
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    img = mmcv.imread(np.array(img))
    return img 
def save_file_st(path_file):
    file_name = path_file.name
    img = load_image(path_file)
    im = Image.fromarray(img)
    output_filename = r'C:\Project\Corn\tempDir' + f'\{file_name}'  
    im.save(output_filename)
    return output_filename


print(time.time())
st.title("Замена текстур")
path_photo = st.file_uploader("Upload An Image", type=['png','jpeg','jpg'], accept_multiple_files=False, key=None,
                  help=None, on_change=None, args=None, kwargs=None,
                  disabled=False)
#option2 = st.selectbox(
#    'Выбор текстуры',
#    ('Пол', 'Стены', 'Потолок'))
check_1 =st.checkbox('Пол', value=False)
option_1 = st.selectbox('Выбор текстуры',('Пробка', 'Ламинат'), key = 1)
check_2 =st.checkbox('Стены', value=False)
option_2 = st.selectbox('Выбор текстуры',('Пробка', 'Ламинат'), key = 2)
check_3 =st.checkbox('Потолок', value=False)
option_3 = st.selectbox('Выбор текстуры',('Пробка', 'Ламинат'), key = 3)
result = st.button('Замена текстур пола', key=None, help=None, on_click=None, args=None, kwargs=None, 
           disabled=False)
if result:
    path_file =save_file_st(path_photo)
    if option_1 == 'Пробка':
        path_to_texture_1 = r'C:\Project\Corn\textures\1.jpg'
    elif option_1 == 'Ламинат':
        path_to_texture_1 = r'C:\Project\Corn\textures\2.jpg'
    if option_2 == 'Пробка':
        path_to_texture_2 = r'C:\Project\Corn\textures\1.jpg'
    elif option_2 == 'Ламинат':
        path_to_texture_2 = r'C:\Project\Corn\textures\2.jpg'
    if option_3 == 'Пробка':
        path_to_texture_3 = r'C:\Project\Corn\textures\1.jpg'
    elif option_3 == 'Ламинат':
        path_to_texture_3 = r'C:\Project\Corn\textures\2.jpg'
    _, _, seg_all, image = moduls.model.predict_and_visualize(model, path_file)
    image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if check_1:
        image_1 = texture.texture_perspective(image_1, path_to_texture_1, seg_all[0])
    if check_2:
        image_1 = texture.texture_perspective(image_1, path_to_texture_2, seg_all[1])
    if check_3:
        image_1 = texture.texture_perspective(image_1, path_to_texture_3, seg_all[2])

    st.image(image_1, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
