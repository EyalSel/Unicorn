In additions to the instructions in assets/install.md, also note that:
 - need to explicitly do pip install Cython
 - Not all mmdet versions work with the required mmcv version, example of verison that works: pip install mmdet==2.25.0

Download the MOT models following the instructions in assets/model_zoo.md


Note that CUDA_VISIBLE_DEVICES=0 must explicitly precede the custom_script.py

