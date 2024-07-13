Для установки среды нужно:
1. Cuda 11.8
2. Cudnn 8.9.7.29
3. Python 3.9
Установка среды черезе VSCode:
1. В терминале VSCode вводим команду python -m [Название среды] venv
2. Активируем созданную среду командой .\verv\Script\activate
Устанавливаем библиотеки:
1. pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
2. pip install openmim
3. mim install mmcv-full==1.6.0
4. mim install mmengine==0.8.0
5. mim install mmsegmentation==0.29.0
Для работы ноутбука установите еще jupyter

Установка streamlit:
1. pip install streamlit==1.11.1
2. Открыть файл по относительному пути .venv\Lib\site-packages\streamlit\elements\arrow_altair.py
3. Исправить на 27 строке from altair.vegalite.v4.api import Chart на from altair.vegalite.v5.api import Chart
