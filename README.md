# Style_Migration_For_Artistic_Font_With_CNN
基于卷积神经网络的风格迁移，制造出带有艺术风格的字体
===
使用方法很简单，只需要输入：<br>
python.exe neural_style_transfer.py 风格图片所在目录  输出文件夹 <br> --chars 花  # 要生成的文字，支持一次输入多个文字
<br>--pictrue_size 300  # 生成图片大小
<br>--background_color (0,0,0)   # 文字图片中背景的颜色
<br>--text_color (255,255,255)   # 文字图片中文字的颜色
<br>--iter 50   # 迭代次数，一般50代左右就行
<br>--smooth_times 20   # 文字图片是否进行模糊化处理
<br>--noise True   # 文字图片是否加入随机噪声
<br>--image_enhance True    # 生成图片是否进行增强，包括色度，亮度，锐度增强
<br>--font_name  宋体  # 文字字体，支持宋体，楷体，黑体，仿宋，等线

