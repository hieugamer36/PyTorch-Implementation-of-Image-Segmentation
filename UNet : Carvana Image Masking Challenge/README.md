# Carvana Image Masking Challenge-UNet Implementation with PyTorch
<h2>Dataset and description of the problem are get from Kaggle : </h2>
https://www.kaggle.com/c/carvana-image-masking-challenge

<h2>UNet paper could be found here :</h2>
https://arxiv.org/abs/1505.04597
<h2>Run train.py for training again the model, for example:</h2>

`python train.py --img_dir="D:/datasets/carvana_image_masking_challenge/train/" --mask_dir="D:/datasets/carvana_image_masking_challenge/train_masks/" --resized_height=128 --resized_width=128 --batchsize=32 --lr=0.0001 --num_workers=4 --epochs=3`

<h2>Training and validation results: </h2>
  <ul>
    <li>num_training : 4579</li>
    <li>num_val : 509 </li>
    <li>learning rate : 1e-4</li>
    <li>resized_height,resized_width : 128</li>
    <li>batch_size: 32</li>
    <li>epochs: 3</li>
  </ul>
  <p>Results :</p>
  <ul>
  <li>train loss ,val loss : 0.12</li>
  <li>train acc ,val acc : 99.24% </li>
  <li>train dice score,val dice score : 0.98 </li>
  </ul>

<h2>Testing on 10 random images from test set</h2>
<h3>Original Images</h3>
<p float="left">
  <img src="./test_results/imgs/6125.jpg" width="128" />
  <img src="./test_results/imgs/9665.jpg" width="128"/>
  <img src="./test_results/imgs/19240.jpg" width="128"/>
  <img src="./test_results/imgs/22269.jpg" width="128"/>
  <img src="./test_results/imgs/24771.jpg" width="128"/>
  <img src="./test_results/imgs/27537.jpg" width="128"/>
  <img src="./test_results/imgs/35536.jpg" width="128"/>
  <img src="./test_results/imgs/55639.jpg" width="128"/>
  <img src="./test_results/imgs/58176.jpg" width="128"/>
  <img src="./test_results/imgs/60608.jpg" width="128"/>
</p>
<h3>Masks</h3>
<p float="left">
  <img src="./test_results/masks/6125.jpg" width="128" />
  <img src="./test_results/masks/9665.jpg" width="128"/>
  <img src="./test_results/masks/19240.jpg" width="128"/>
  <img src="./test_results/masks/22269.jpg" width="128"/>
  <img src="./test_results/masks/24771.jpg" width="128"/>
  <img src="./test_results/masks/27537.jpg" width="128"/>
  <img src="./test_results/masks/35536.jpg" width="128"/>
  <img src="./test_results/masks/55639.jpg" width="128"/>
  <img src="./test_results/masks/58176.jpg" width="128"/>
  <img src="./test_results/masks/60608.jpg" width="128"/>
</p>
<h3>Combine masks into images</h3>
<p float="left">
  <img src="./test_results/combinations/6125.jpg" width="128" />
  <img src="./test_results/combinations/9665.jpg" width="128"/>
  <img src="./test_results/combinations/19240.jpg" width="128"/>
  <img src="./test_results/combinations/22269.jpg" width="128"/>
  <img src="./test_results/combinations/24771.jpg" width="128"/>
  <img src="./test_results/combinations/27537.jpg" width="128"/>
  <img src="./test_results/combinations/35536.jpg" width="128"/>
  <img src="./test_results/combinations/55639.jpg" width="128"/>
  <img src="./test_results/combinations/58176.jpg" width="128"/>
  <img src="./test_results/combinations/60608.jpg" width="128"/>
</p>
