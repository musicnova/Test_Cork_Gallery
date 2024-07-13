import os
import torch
import mmcv
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv import Config

repo_url = 'https://github.com/open-mmlab/mmsegmentation/archive/refs/heads/master.zip'
repo_dir = 'mmsegmentation-master'
config_dir = os.path.join(repo_dir, 'configs')

def edit_model():
# Скачиваем и распаковываем репозиторий, если он еще не скачан
    # Получаем список конфигурационных файлов
    config_path = r'C:\Project\Corn\config\upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py'
    checkpoint_path = r'C:\Project\Corn\upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'
    # Загружаем модель один раз
    cfg = Config.fromfile(config_path)
    cfg.model.test_cfg.avg_non_ignore = True
    if 'evaluation' not in cfg:
        cfg.evaluation = dict(metric=['mIoU', 'mDice', 'mAcc', 'precision', 'recall'])
    model = init_segmentor(cfg, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    return model
def predict_and_visualize(model, image_path, dataset_name = 'ade20k'):
    # Предсказание
    color_mapping = {
        'ade20k': {0: (100), 3: (150), 5: (200)}
    }
    image = mmcv.imread(image_path)
    result = inference_segmentor(model, image)

    # Преобразование тензоров в numpy массивы
    seg_map = result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0]
    seg_floor = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_floor[seg_map == 3, :] = 150
    seg_ceiling = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_ceiling[seg_map == 5, :] = 200
    seg_wall  = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_wall[seg_map == 0, :] = 100
    seg_all = [seg_floor, seg_wall, seg_ceiling]
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in color_mapping[dataset_name].items():
        color_seg[seg_map == label, :] = color

    img = image * 0.7 + color_seg * 0.3
    img = img.astype(np.uint8)

    return seg_map, color_seg, seg_all, image
