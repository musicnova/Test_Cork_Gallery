Для установки среды нужно:
1. Cuda 11.8
2. Cudnn 8.9.7.29
3. Python 3.9 (только не 3.9.7)
Установка среды черезе VSCode:
1. В терминале VSCode вводим команду python -m [Название среды] venv
2. Активируем созданную среду командой .\verv\Script\activate
Устанавливаем библиотеки:
1. pip uninstall numpy
2. pip install numpy==1.26.4
1. pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
2. pip install openmim
3. mim install mmcv-full==1.6.0
4. mim install mmengine==0.8.0
5. mim install mmsegmentation==0.29.0
6. pip install streamlit
7. pip install streamlit_image_select
8. pip install streamlit_image_annotation
9. pip install streamlit_image_comparison
10. pip install scipy


Далее нужно заменить следующие файлы из архива "файлы для замены":

1. .venv\Lib\site-packages\streamlit_image_select\frontend\build\styles.css
2. .venv\Lib\site-packages\streamlit_image_annotation\Point\frontend\build\static\js\main.d4738267.js

Далее в файле settings.py нужно поправить пути.
path_icon = r'.\textures\tex_icon'
path_json = r'.\textures\tex_json'
path_logo = r'.\logo\logo.png'
config_path = r'.\config\upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py'
checkpoint_path = r'.\upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'
Ссылка на google диск
https://drive.google.com/drive/folders/13c6xBqIIyukCMydDyo57QziqQOu5OjhD?usp=sharing
