# Image Style Transfer

This is the academic research project for studying Image Style Transfer and effects of changing the architecture of VGG and transform networks.



# Requirements

* Please download [Pretrained VGG network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
* Set following paths in the 'Image_style_transfer_final_project.py' or Image_style_transfer_final_project.ipynb' file pointing to all the directories and file locations.

```
MODEL_SAVE_PATH = './saved_model/fns.ckpt'
VGG_PATH = './imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = './train_sample/'
STYLE_PATH = './style/wave.jpg'

TEST_IMAGE_PATH = './test_images/stata.jpg'
OUTPUT_PATH = './output/'
```

### Project members
* Aravinda Misra
* Saurabh Deshmukh 
* [Tanmay Bhatt](https://www.github.com/TanmayAB)

## Attributions/Thanks
* The project also borrowed some code (in VGG network) from Anish's [Neural Style](https://github.com/anishathalye/neural-style/)
* We also refereed lengstorm's [Image Style Transfer](https://github.com/lengstrom/fast-style-transfer) for Instance normalization approach and loss optimization.