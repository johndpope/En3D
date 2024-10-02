import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


image_control_3d_portrait = pipeline(
    Tasks.image_control_3d_portrait,
    model='damo/cv_vit_image-control-3d-portrait-synthesis',
)

test_image = './assets/characters/000243.png'
save_dir = './results'
os.makedirs(save_dir, exist_ok=True)

image_control_3d_portrait(
    dict(image=test_image, save_dir=save_dir))
print('finished!')
### render results will be saved in save_dir