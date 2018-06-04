# Style Migration For Artistic Font With CNN
基于卷积神经网络的风格迁移，制造出带有艺术风格的字体
===
使用方法很简单，只需要输入：<br>
python.exe neural_style_transfer.py 风格图片所在目录  输出文件夹 
<br>  --chars 花  # 要生成的文字，支持一次输入多个文字
<br>  --pictrue_size 300  # 生成图片大小
<br>  --background_color (0,0,0)   # 文字图片中背景的颜色
<br>  --text_color (255,255,255)   # 文字图片中文字的颜色
<br>  --iter 50   # 迭代次数，一般50代左右就行
<br>  --smooth_times 20   # 文字图片是否进行模糊化处理
<br>  --noise True   # 文字图片是否加入随机噪声
<br>  --image_enhance True    # 生成图片是否进行增强，包括色度，亮度，锐度增强
<br>  --font_name  宋体  # 文字字体，支持宋体，楷体，黑体，仿宋，等线
<br>  --reverse_color False  # True-黑纸白字，False-白纸黑字，默认白纸黑字


## 一些说明
神经网络基于[keras](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)官方的风格迁移的例子，经过一定的调整loss函数和调参后得到最适合于艺术风格字体的图片。<br>
调整包括：<br>
##### 1.加入生成文字图片的方法，以及提供一系列图片生成相关接口，便于您第一时间修改结果
##### 2.修改了loss函数，经过大量实验，确定使用keras提供的VGG19网络的'block1_conv1','block2_conv1','block3_conv1'三层作为风格损失，去除内容损失
##### 3.加入一些图片的增强方法，使得结果更加色彩丰富
##### 4.在style文件夹下提供了一系列图片供您探索
##### 4.运行需要Keras支持，建议使用GPU运算，在	Nvidia GeForce GTX 1050 Ti (4 GB)上，一次迭代大约3s，一张图片耗时2-3min

	
## 下面给出一些例子
![竹](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E7%AB%B9_%E4%BB%A3%E6%95%B0_49.png)  
![花](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E8%8A%B1_%E4%BB%A3%E6%95%B0_49.png)  
![雨](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E9%9B%A8_%E4%BB%A3%E6%95%B0_49.png)  
![雾](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E9%9B%BE_%E4%BB%A3%E6%95%B0_49.png)  
![墨](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E5%A2%A8_%E4%BB%A3%E6%95%B0_49.png)  
![木](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E6%9C%A8_%E4%BB%A3%E6%95%B0_49.png)  
![火](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E7%81%AB_%E4%BB%A3%E6%95%B0_49.png)  
![星](https://github.com/yuweiming70/Style_Migration_For_Artistic_Font_With_CNN/blob/master/example/%E6%98%9F_%E4%BB%A3%E6%95%B0_49.png)  
