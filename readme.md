#  DeblurrGAN : Blind Motion Deblurring Using Conditional Adversarial Networks
An implementation of DeblurrGAN described in the paper using tensorflow.
* [ DeblurrGAN : Blind Motion Deblurring Using Conditional Adversarial Networks](https://arxiv.org/abs/1711.07064)

Published in CVPR 2018, written by O. Kupyn, V. Budzan, M. Mykhailych, D. Mishkin and J. Matas

## Requirement
- Python 3.6.5
- Tensorflow 1.10.1 
- Pillow 5.0.0
- numpy 1.14.5
- Pretrained VGG19 file : [vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) (for training!)

## Datasets
- [GOPRO dataset](https://github.com/SeungjunNah/DeepDeblur_release)

## Pre-trained model
- [GOPRO_model](https://drive.google.com/open?id=1Sg0LQUAsf3wfDQNMwUKKM2O-uJMWQWxw)

## Experimental Results
Experimental results on GOPRO dataset

| Blur | Sharp | Ground Truth |
| --- | --- | --- |
| <img src="images/blur/blur_00.png" width="640px"> |<img src="images/result/result_00.png" width="640px"> | <img src="images/sharp/sharp_00.png" width="640px"> |
| <img src="images/blur/blur_01.png" width="640px"> |<img src="images/result/result_01.png" width="640px"> | <img src="images/sharp/sharp_01.png" width="640px"> |
| <img src="images/blur/blur_02.png" width="640px"> |<img src="images/result/result_02.png" width="640px"> | <img src="images/sharp/sharp_02.png" width="640px"> |
| <img src="images/blur/blur_03.png" width="640px"> |<img src="images/result/result_03.png" width="640px"> | <img src="images/sharp/sharp_03.png" width="640px"> |
| <img src="images/blur/blur_04.png" width="640px"> |<img src="images/result/result_04.png" width="640px"> | <img src="images/sharp/sharp_04.png" width="640px"> |
| <img src="images/blur/blur_05.png" width="640px"> |<img src="images/result/result_05.png" width="640px"> | <img src="images/sharp/sharp_05.png" width="640px"> |

## Comments
If you have any questions or comments on my codes, please email to me. [son1113@snu.ac.kr](mailto:son1113@snu.ac.kr)

## Reference
[1] https://github.com/KupynOrest/DeblurGAN