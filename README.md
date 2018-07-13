# Style Migration For Artistic Font With CNN
基于卷积神经网络的风格迁移，超越艺术字
===
使用方法很简单，只需要输入：<br>
python.exe neural_style_transfer.py "风格图片所在目录"  "输出文件夹"
<br>  --chars 花  # 要生成的文字，支持一次输入多个文字
<br>  --pictrue_size 300  # 生成图片大小
<br>  --background_color (0,0,0)   # 文字图片中背景的颜色
<br>  --text_color (255,255,255)   # 文字图片中文字的颜色
<br>  --iter 50   # 迭代次数，一般50代左右就行
<br>  --smooth_times 20   # 文字图片是否进行模糊化处理
<br>  --noise 10   # 文字图片加入随机噪声的等级
<br>  --image_enhance True    # 生成图片是否进行增强，包括色度，亮度，锐度增强
<br>  --font_name  宋体  # 文字字体，支持宋体，楷体，黑体，仿宋，等线
<br>  --reverse_color False  # True-黑纸白字，False-白纸黑字，默认白纸黑字
<br>  --output_per_iter 2  # 每隔多少次迭代输出一张图片
<br>  --image_input_mode  one_pic  # 输入的风格图片允许使用一下mode： 'one_pic:一张风格图片'，'one_pic_T:一张风格图片，但是这张图片经过旋转90度后当作第二张,特别适合汉字的横竖笔画'，'two_pic:两张风格图片'
<br>  --style_reference_image2_path  # 第二张风格图片的位置，没有第二张不填
<br>  --two_style_k 0.9  # 两张图片的相对权重，第一张*k+第二张*(1-k)

## 一些使用的例子
单一风格：<br>
"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "E:\学习\大三下\人工智能导论\风格转移字体\neural_style_transfer.py"  "E:\学习\大三下\人工智能导论\风格转移字体\style\bamboo\5.jpg" E:\学习\大三下\人工智能导论\风格转移字体\输出\ --pictrue_size 300 --background_color (255,255,255) --text_color (0,0,0) --iter 30 --chars 竹 --smooth_times 20 --noise 10 --image_enhance True --image_input_mode one_pic

两个风格：<br>
"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "E:\学习\大三下\人工智能导论\风格转移字体\neural_style_transfer.py"  "E:\学习\大三下\人工智能导论\风格转移字体\style\stone\2.jpg" E:\学习\大三下\人工智能导论\风格转移字体\输出\ --pictrue_size 300 --background_color (255,255,255) --text_color (0,0,0) --iter 50 --chars 石 --smooth_times 20 --noise  10 --image_enhance True --image_input_mode two_pic --style_reference_image2_path "E:\学习\大三下\人工智能导论\风格转移字体\style\stone\3.jpg" --two_style_k 0.6

# 7.13更新
##### 1.更新了使用例子
##### 2.代码稍作调整

## 6.13更新
##### 1.支持两张风格图片，使用第一张*k+第二张*(1-k)，可以平滑调节两张图片的风格过渡。
##### 2.支持调节随机噪音的强度，为图片加上“noise×图片边长”个噪点。
##### 3.精细调参，三层卷积层的权重改为10:1:1，颜色更艳丽。

## 一些说明
神经网络基于[keras](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)官方的风格迁移的例子，经过一定的调整loss函数和调参后得到最适合于艺术风格字体的代码。<br>
调整包括：<br>
##### 1.加入生成文字图片的方法，以及提供一系列图片生成相关接口，便于您第一时间修改结果
##### 2.修改了loss函数，经过大量实验，确定使用keras提供的VGG19网络的'block1_conv1','block2_conv1','block3_conv1'三层作为风格损失，去除内容损失
##### 3.加入一些图片的增强方法，使得结果更加色彩丰富
##### 4.在style文件夹下提供了一系列图片供您探索
##### 4.运行需要Keras支持，建议使用GPU运算，在	Nvidia GeForce GTX 1050 Ti (4 GB)上，一次迭代大约3s，一张图片耗时2-3min

	
## 下面给出一些例子，在example文件夹下有其对应的风格图片
![花](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E8%8A%B1_%E4%BB%A3%E6%95%B0_49.png)  
![雨](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E9%9B%A8_%E4%BB%A3%E6%95%B0_49.png)  
![竹](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E7%AB%B9_%E4%BB%A3%E6%95%B0_49.png)  
![雾](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E9%9B%BE_%E4%BB%A3%E6%95%B0_49.png)  
![墨](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E5%A2%A8_%E4%BB%A3%E6%95%B0_49.png)  
![木](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E6%9C%A8_%E4%BB%A3%E6%95%B0_49.png)  
![火](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E7%81%AB_%E4%BB%A3%E6%95%B0_49.png)  
![星](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E6%98%9F_%E4%BB%A3%E6%95%B0_49.png)  
![石](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E7%9F%B3_%E4%BB%A3%E6%95%B0_150.png)  

