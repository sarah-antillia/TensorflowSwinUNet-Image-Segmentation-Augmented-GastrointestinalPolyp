<h2>TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp (2024/03/13)</h2>

This is the second experimental Image Segmentation project for GastrointestinalPolyp  based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1f9Bwjt2rKstDRgv5zztfrBl-DLZNo--c/view?usp=sharing">
GastrointestinalPolyp-ImageMask-Dataset.zip</a> (2024/03/12)
<br>
<br>
Segmentation samples.<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/segmentation_samples.png" width="720" height="auto">
<br>
<br>

We will use an online dataset augmentation strategy based on Python script <a href="./src/ImageMaskAugmentor.py">
ImageMaskAugmentor.py</a> to train aGastrointestinalPolyp Segmentation Model.<br><br>
Please see also our first experiment 
<a href="https://github.com/atlan-antillia/Image-Segmentation-Gastrointestinal-Polyp">Image-Segmentation-Gastrointestinal-Polyp</a>
<br>
<br>
We use the SwinTransformer UNet Model
<a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a> for this GastrointestinalPolyp Segmentation.<br>

<h3>1. Dataset Citatioin</h3>

The image dataset used here has been taken from the following kaggle web site.<br>
<a href="https://www.kaggle.com/datasets/debeshjha1/kvasirseg">Kvasir-SEG Data (Polyp segmentation & detection)</a>

<b>About Dataset</b>
<pre>
Kvasir-SEG information:
The Kvasir-SEG dataset (size 46.2 MB) contains 1000 polyp images and their corresponding ground truth 
from the Kvasir Dataset v2. The images' resolution in Kvasir-SEG varies from 332x487 to 1920x1072 pixels. 
The images and its corresponding masks are stored in two separate folders with the same filename. 
The image files are encoded using JPEG compression, facilitating online browsing. 
The open-access dataset can be easily downloaded for research and educational purposes.
</pre>

<b>Applications of the Dataset</b><br>
<pre>
The Kvasir-SEG dataset is intended to be used for researching and developing new and improved methods 
for segmentation, detection, localization, and classification of polyps. 
Multiple datasets are prerequisites for comparing computer vision-based algorithms, and this dataset 
is useful both as a training dataset or as a validation dataset. These datasets can assist the 
development of state-of-the-art solutions for images captured by colonoscopes from different manufacturers. 
Further research in this field has the potential to help reduce the polyp miss rate and thus improve 
examination quality. The Kvasir-SEG dataset is also suitable for general segmentation and bounding box 
detection research. In this context, the datasets can accompany several other datasets from a wide 
range of fields, both medical and otherwise.
</pre>
<!--
<b>Ground Truth Extraction</b><br>
<pre>
We uploaded the entire Kvasir polyp class to Labelbox and created all the segmentations using this application. 
The Labelbox is a tool used for labeling the region of interest (ROI) in image frames, i.e., the polyp regions 
for our case. We manually annotated and labeled all of the 1000 images with the help of medical experts. 
After annotation, we exported the files to generate masks for each annotation. 
The exported JSON file contained all the information about the image and the coordinate points for generating 
the mask. To create a mask, we used ROI coordinates to draw contours on an empty black image and fill the 
contours with white color. The generated masks are a 1-bit color depth images. The pixels depicting polyp tissue, 
the region of interest, are represented by the foreground (white mask), while the background (in black) does not 
contain positive pixels. Some of the original images contain the image of the endoscope position marking probe, 
ScopeGuide TM, Olympus Tokyo Japan, located in one of the bottom corners, seen as a small green box. 
As this information is superfluous for the segmentation task, we have replaced these with black boxes in the 
Kvasir-SEG dataset.
</pre>
-->
See also:
<pre>
Kvasir-SEG
https://paperswithcode.com/dataset/kvasir-seg
</pre>


<h3>
<a id="2">
2 GastrointestinalPolyp-ImageMask Dataset
</a>
</h3>
 If you would like to train this GastrointestinalPolyp Segmentation model by yourself,
please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1f9Bwjt2rKstDRgv5zztfrBl-DLZNo--c/view?usp=sharing">
GastrointestinalPolyp-ImageMask-Dataset.zip</a>.
<br>


Please see also the <a href="https://github.com/sarah-antillia/ImageMask-Dataset-GastrointestinalPolyp">ImageMask-Dataset-GastrointestinalPolyp</a>.<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be

<pre>
./dataset
└─GastrointestinalPolyp
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
 
 
<b>GastrointestinalPolyp Dataset Statistics</b><br>
<img src ="./projects/TensorflowSwinUNet/GastrointestinalPolyp/GastrointestinalPolyp_Statistics.png" width="512" height="auto"><br>
As shown above, the number of images of train and valid dataset is not necessarily large. 
<br>

<h3>
<a id="3">
3 TensorflowSwinUNet
</a>
</h3>
This <a href="./src/TensorflowUNet.py">TensorflowUNet</a> model is slightly flexibly customizable by a configuration file.<br>
For example, <b>TensorflowSwinUNet/GastrointestinalPolyp</b> model can be customizable
by using <a href="./projects/TensorflowSwinUNet/GastrointestinalPolyp/train_eval_infer_augmentor.config">train_eval_infer_augmentor.config</a>
<pre>
; train_eval_infer_augmentor.config
; 2024/03/12 (C) antillia.com

[model]
model          = "TensorflowSwinUNet"
generator      = True
image_width    = 512
image_height   = 512
activation     = "mish"
optimizer      = "Adam"

image_channels = 3
num_classes    = 1
;filter_num_begin = 128
filter_num_begin = 64

; number of channels in the first downsampling block; it is also the number of embedded dimensions

depth = 4
; the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 

stack_num_down = 2         
; number of Swin Transformers per downsampling level

stack_num_up = 2
; number of Swin Transformers per upsampling level

patch_size = (4, 4)        
; Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.  

num_heads = [4, 8, 8, 8]   
;number of attention heads per down/upsampling level

window_size = [4, 2, 2, 2] 
;the size of attention window per down/upsampling level

num_mlp = 512              
; number of MLP nodes within the Transformer

shift_window=True          
;Apply window shifting, i.e., Swin-MSA

learning_rate  = 0.0001
clipvalue      = 0.2
loss           = "bce_dice_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_CUBIC"

[train]
save_model_file = ""

dataset_splitter = True
learning_rate_reducer = True
reducer_patience      = 5

epochs        = 100
batch_size    = 2
patience      = 10
metrics       = ["binary_accuracy", "val_binary_accuracy"]
;metrics       = ["iou_coef", "val_iou_coef"]
model_dir     = "./models"
eval_dir      = "./eval"

image_datapath = "../../../dataset/GastrointestinalPolyp/train/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/train/masks/"
create_backup  = False

[eval]
image_datapath = "../../../dataset/GastrointestinalPolyp/valid/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/valid/masks/"

[test]
image_datapath = "../../../dataset/GastrointestinalPolyp/test/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/test/masks/"

[infer] 
images_dir   = "../../../dataset/GastrointestinalPolyp/test/images/"
output_dir   = "./test_output"
merged_dir   = "./test_output_merged"

[tiledinfer] 
overlapping  = 64
images_dir   = "../../../dataset/GastrointestinalPolyp/test/images/" 
output_dir   = "./tiled_test_output"
merged_dir   ="./tiled_test_output_merged"

[mask]
blur        = True
binarize    = True
threshold   = 128

[generator]
debug        = True
augmentation = True

[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [5, 10,]
shrinks  = [0.8]
shears   = [0.2]
transformer = True
alpah       = 1300
sigmoid     = 8
</pre>

Please note that the online augementor 
<a href="./src/ImageMaskAugmentor.py">
ImageMaskAugmentor.py</a> reads the parameters in [generator] and [augmentor] sections, and yields some images and mask depending on the batch_size,
 which are used for each epoch of the training and evaluation process of this UNet Model. 
<pre>
[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [5, 10,]
shrinks  = [0.8]
shears   = [0.2]
transformer = True
alpah       = 1300
sigmoid     = 8

</pre>
Depending on these parameters in [augmentor] section, it will generate vflipped, hflipped, rotated, shrinked,
sheared, elastic-transformed images and masks
from the original images and masks in the folders specified by image_datapath and mask_datapath in 
[train] and [eval] sections.<br>
<pre>
[train]
image_datapath = "../../../dataset/GastrointestinalPolyp/train/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/train/masks/"
[eval]
image_datapath = "../../../dataset/GastrointestinalPolyp/valid/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/valid/masks/"
</pre>

For more detail on ImageMaskAugmentor.py, please refer to
<a href="https://github.com/sarah-antillia/Image-Segmentation-ImageMaskDataGenerator">
Image-Segmentation-ImageMaskDataGenerator.</a>.
    
<br>

<h3>
3.1 Training
</h3>
Please move to a <b>./projects/TensorflowSwinUNet/GastrointestinalPolyp</b> folder,<br>
and run the following bat file to train TensorflowUNet model for GastrointestinalPolyp.<br>
<pre>
./1.train_generator.bat
</pre>
, which simply runs <a href="./src/TensorflowUNetGeneratorTrainer.py">TensorflowUNetGeneratorTrainer.py </a>
in the following way.

<pre>
python ../../../src/TensorflowUNetGeneratorTrainer.py ./train_eval_infer_augmentor.config
</pre>
Train console output:<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/train_console_output_at_epoch_52.png" width="720" height="auto"><br>
Train metrics:<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/train_metrics.png" width="720" height="auto"><br>
Train losses:<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/train_losses.png" width="720" height="auto"><br>
<br>
The following debug setting is helpful whether your parameters in [augmentor] section are good or not good.
<pre>
[generator]
debug     = True
</pre>
You can check the yielded images and mask files used in the actual train-eval process in the following folders under
<b>./projects/TensorflowSwinUNet/GastrointestinalPolyp/</b>.<br> 
<pre>
generated_images_dir
generated_masks_dir
</pre>

Sample images in generated_images_dir<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/sample_images_in_generated_images_dir.png"
 width="1024" height="auto"><br>
Sample masks in generated_masks_dir<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/sample_masks_in_generated_masks_dir.png"
 width="1024" height="auto"><br>

<h3>
3.2 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSwinUNet/GastrointestinalPolyp</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for GastrointestinalPolyp.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/evaluate_console_output_at_epoch_52.png" width="720" height="auto">

<pre>
Test loss    :0.338
Test accuracy:0.9052000045776367
</pre>
As shown above, the loss score for the test dataset is far from better than that of the first experiment based on
TensorflowSlightlyFlexibleUNet Model
<a href="https://github.com/atlan-antillia/Image-Segmentation-Gastrointestinal-Polyp">Image-Segmentation-Gastrointestinal-Polyp</a>
<br>
<img src="https://github.com/atlan-antillia/Image-Segmentation-Gastrointestinal-Polyp/blob/main/asset/evaluate_console_at_epoch_50_0623.png"><br> 

<br>
<h2>


3.3 Inference
</h2>
Please move to a <b>./projects/TensorflowSwinUNet/GastrointestinalPolyp</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for GastrointestinalPolyp.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer.config
</pre>
Sample test images<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/sample_test_images.png" width="1024" height="auto"><br>
Sample test mask (ground_truth)<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/sample_test_masks.png" width="1024" height="auto"><br>

<br>
Inferred test masks<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/test_output.png" width="1024" height="auto"><br>
<br>
Merged test images and inferred masks<br> 
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/asset/test_output_merged.png" width="1024" height="auto"><br> 

<!--
Enlarged samples<br>
<table>
<tr>
<td>
test/images/0003-l.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/0003-l.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/10040_HFH_017.jpg<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/test_output_merged/0003-l.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
test/images/0031-s.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/0031-s.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/0031-s.jpg<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/test_output_merged/0031-s.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/0079-l.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/0079-l.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/0079-l.jpg<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/test_output_merged/0079-l.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/0112-l.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/0112-l.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/0112-l.jpg<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/test_output_merged/0112-l.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/0852-s.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/0852-s.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/0852-s.jpg<br>
<img src="./projects/TensorflowSwinUNet/GastrointestinalPolyp/test_output_merged/0852-s.jpg" width="512" height="auto">
</td> 
</tr>

</table>
-->

<h3>
References
</h3>

<b>1. Kvasir-SEG Data (Polyp segmentation & detection)</b><br>
<pre>
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>

<b>2. Kvasir-SEG: A Segmented Polyp Dataset</b><br>
Debesh Jha, Pia H. Smedsrud, Michael A. Riegler, P˚al Halvorsen,<br>
Thomas de Lange, Dag Johansen, and H˚avard D. Johansen<br>
<pre>
https://arxiv.org/pdf/1911.07069v1.pdf
</pre>

<b>3. Image-Segmentation-GastrointestinalPolyp</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/atlan-antillia/Image-Segmentation-GastrointestinalPolyp

</pre>

