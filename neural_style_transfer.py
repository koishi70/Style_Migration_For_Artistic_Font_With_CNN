from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from scipy.misc import imsave
from keras.applications import vgg19
from keras import backend as K
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter

import random
random.seed(0)

def save_img(fname, image, image_enhance=False):  # 图像可以增强
    image = Image.fromarray(image)
    if image_enhance:
        # 亮度增强
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.2
        image = enh_bri.enhance(brightness)

        # 色度增强
        enh_col = ImageEnhance.Color(image)
        color = 1.2
        image = enh_col.enhance(color)

        # 锐度增强
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = 1.2
        image = enh_sha.enhance(sharpness)
    imsave(fname, image)
    return


def smooth(image):  # 模糊图片
    w, h, c = image.shape
    smoothed_image = np.zeros([w - 2, h - 2,c])
    smoothed_image += image[:w - 2, 2:h,:]
    smoothed_image += image[1:w-1, 2:,:]
    smoothed_image += image[2:, 2:h,:]
    smoothed_image += image[:w-2, 1:h-1,:]
    smoothed_image += image[1:w-1, 2:h,:]
    smoothed_image += image[2:, 1:h - 1,:]
    smoothed_image += image[:w-2, :h-2,:]
    smoothed_image += image[1:w-1, :h - 2,:]
    smoothed_image += image[2:, :h - 2,:]
    smoothed_image /= 9.0
    return smoothed_image.astype("uint8")


def str_to_tuple(s):
    s = list(s)
    ans = list()
    temp = ""
    for i in range(len(s)):
        if s[i] == '(' :
            continue
        if s[i] == ',' or s[i] == ')':
            ans.append(int(temp))
            temp = ""
            continue
        temp += s[i]
    return tuple(ans)


def char_to_picture(text="", font_name="宋体", background_color=(255,255,255), text_color=(0,0,0), pictrue_size=400,
                    text_position=(0, 0), in_meddium=False, reverse_color=False,smooth_times=0,noise=0):
    pictrue_shape = (pictrue_size,pictrue_size)
    im = Image.new("RGB", pictrue_shape, background_color)
    dr = ImageDraw.Draw(im)

    # 由于系统内部不是使用汉字文件名，而是英文名，在此转换
    if font_name == "宋体":
        font_name = "SIMSUN.ttc"
    if font_name == "楷体":
        font_name = "SIMKAI.ttf"
    if font_name == "黑体":
        font_name = "SIMHEI.ttf"
    if font_name == "等线":
        font_name = "DENG.ttf"
    if font_name == "仿宋":
        font_name = "SIMFANG.ttf"

    # 取得字体文件的位置
    font_dir = "C:\Windows\Fonts\\" + font_name
    font_size = int(pictrue_size * 0.8 / len(text)) # 设定文字的大小
    font = ImageFont.truetype(font_dir, font_size)

    # 开始绘图
    # 如果设置了居中，那么就居中
    # 英文字母的对齐方式并不一样
    char_dict = []
    for i in range(26):
        char_dict.append(chr(i + ord('a')))
        char_dict.append(chr(i + ord('A')))
    if in_meddium:
        char_num = len(text)
        text_position = (pictrue_shape[0]/2 - char_num*font_size/2, pictrue_shape[1]/2 - font_size/2)  # 中文
        if text in char_dict:
            text_position = (pictrue_shape[0] / 2 - char_num*font_size/4, pictrue_shape[1] / 2 - font_size / 2)  # 英文


    # 开始绘制图像
    dr.text(text_position, text, font=font, fill=text_color)
    if reverse_color:
        im = ImageOps.invert(im)

    # 随机扰动
    if noise > 0:
        print("adding noise...")
        im_array = np.array(im)
        noise_num = noise * pictrue_size
        for i in range(noise_num):
            pos = (random.randint(0,pictrue_size-1), random.randint(0,pictrue_size-1))
            color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
            im_array[pos[0],pos[1],:] = color
        im = Image.fromarray(im_array)

    # 模糊化图片
    '''
    for i in range(smooth_times):
        im =im.filter(ImageFilter.GaussianBlur)
    '''
    im_array = np.array(im)
    for i in range(smooth_times):
        im_array = smooth(im_array)
    im = Image.fromarray(im_array)

    # 图片经过模糊后略有缩小
    im = im.resize(pictrue_shape)
    print("文字转换图片成功")
    return im

# 输入参数
parser = argparse.ArgumentParser(description='基于Keras的风格迁移字体.')  # 解析器
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='风格图片的位置')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='保存结果图片的前缀')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='迭代次数')
parser.add_argument('--chars', type=str, default="花", required=False,
                    help='输入要转换的文字.')
parser.add_argument('--reverse_color', type=bool, default=False, required=False,
                    help='True-黑纸白字，False-白纸黑字，默认白纸黑字.')
parser.add_argument('--pictrue_size', type=int, default=400, required=False,
                    help='图片大小.')
parser.add_argument('--font_name', type=str, default="宋体", required=False,
                    help='文字字体.')
parser.add_argument('--smooth_times', type=int, default=0, required=False,
                    help='文字图片是否模糊的强度.')
parser.add_argument('--background_color', type=str, default="(255,255,255)", required=False,
                    help='文字图片背景颜色.')
parser.add_argument('--text_color', type=str, default="(0,0,0)", required=False,
                    help='文字颜色.')
parser.add_argument('--noise', type=int, default=1, required=False,
                    help='加上随机噪音的等级.')
parser.add_argument('--image_enhance', type=bool, default=False, required=False,
                    help='图像增强.')
parser.add_argument('--image_input_mode', type=str, default="one_pic", required=False,
                    help='输入的风格图片允许使用一下mode：'
                         'one_pic:一张风格图片'
                         'one_pic_T:一张风格图片，但是这张图片经过旋转90度后当作第二张,特别适合汉字的横竖笔画'
                         'two_pic:两张风格图片')
parser.add_argument('--two_style_k', type=float, default=0.5, required=False,
                    help='两张图片的相对权重，第一张*k+第二张*(1-k)')
parser.add_argument('--style_reference_image2_path', metavar='ref', type=str, required=False,
                    help='第二张风格图片的位置')

# 获取参数
args = parser.parse_args()
style_reference_image_path = args.style_reference_image_path
style_reference_image2_path = args.style_reference_image2_path
result_prefix = args.result_prefix
iterations = args.iter
chars = args.chars
reverse_color = args.reverse_color
pictrue_size = args.pictrue_size
font_name = args.font_name
smooth_times = args.smooth_times
noise = args.noise
image_enhance = args.image_enhance
background_color = str_to_tuple(args.background_color)
text_color = str_to_tuple(args.text_color)
image_input_mode = args.image_input_mode
two_style_k = args.two_style_k

# 生成输入图片
char_image = char_to_picture(chars,font_name=font_name,background_color=background_color,text_color=text_color,
                             pictrue_size=pictrue_size,in_meddium=True,reverse_color=reverse_color,
                             smooth_times=smooth_times,noise=noise)
width, height = char_image.size

# 风格损失的权重没有意义，因为对于一张文字图片来说，不可能有没有内容损失
style_weight = 1.0

# util function to resize and format pictures into appropriate tensors
def preprocess_image(image):
    """
    预处理图片，包括变形到(1，width, height)形状，数据归一到0-1之间
    :param image: 输入一张图片
    :return: 预处理好的图片
    """
    image = image.resize((width, height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # (width, height)->(1，width, height)
    image = vgg19.preprocess_input(image)  # 0-255 -> 0-1.0
    return image

def deprocess_image(x):
    """
    将0-1之间的数据变成图片的形式返回
    :param x: 数据在0-1之间的矩阵
    :return: 图片，数据都在0-255之间
    """
    x = x.reshape((width, height, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')  # 以防溢出255范围
    return x

# 得到需要处理的数据，处理为keras的变量（tensor），处理为一个(5, width, height, 3)的矩阵
# 分别是文字图片，风格图片1，风格图片1T, 风格图片2，结果图片
base_image = K.variable(preprocess_image(char_image))
style_reference_image1 = K.variable(preprocess_image(load_img(style_reference_image_path)))
style_reference_image1_T = K.variable(preprocess_image(load_img(style_reference_image_path).transpose(Image.ROTATE_90)))
try:
    style_reference_image2 = K.variable(preprocess_image(load_img(style_reference_image2_path)))
except:  # 不会用到这个了
    if image_input_mode == "two_pic":
        print("尚未找到第二张图片，或许您忘记输入了，请输入--style_reference_image2_path 第二张图片的位置")
    style_reference_image2 = K.variable(preprocess_image(load_img(style_reference_image_path)))

combination_image = K.placeholder((1, width, height, 3))
input_tensor = K.concatenate([base_image, style_reference_image1, style_reference_image1_T,
                              style_reference_image2, combination_image], axis=0)
# 结合以上5张图片，作为输入向量

# 使用Keras提供的训练好的Vgg19网络
model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792             A
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856            B
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168           C
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160          D
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808          E
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, None, None, 512)   2359808          F
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, None, None, 512)   0
=================================================================
'''
# Vgg19网络中的不同的名字，储存起来以备使用
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

def gram_matrix(x):  # Gram矩阵
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# 风格损失，是风格图片与结果图片的Gram矩阵之差，并对所有元素求和
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    S_C = S-C
    channels = 3
    size = height * width
    return K.sum(K.square(S_C)) / (4. * (channels ** 2) * (size ** 2))
    #return K.sum(K.pow(S_C,4)) / (4. * (channels ** 2) * (size ** 2))  # 居然和平方没有什么不同
    #return K.sum(K.pow(S_C,4)+K.pow(S_C,2)) / (4. * (channels ** 2) * (size ** 2))  # 也能用，花后面出现了叶子


loss = K.variable(0.)
# 计算风格损失，糅合多个特征层的数据，取平均
#                  [     A,              B,              C,              D,              E,              F     ]
# feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1','block5_conv1','block5_conv4']
#                   A全是颜色，没有纹理---------------------------------------------------->F全是纹理，没有颜色
feature_layers = ['block1_conv1','block2_conv1','block3_conv1']
feature_layers_w = [10.0,1.0,1.0]
for i in range(len(feature_layers)):
    # 每一层的权重以及数据
    layer_name, w = feature_layers[i], feature_layers_w[i]
    layer_features = outputs_dict[layer_name]

    style_reference_features1 = layer_features[1, :, :, :]
    combination_features = layer_features[4, :, :, :]

    if image_input_mode == "one_pic":
        style_reference_features_mix = style_reference_features1
    elif image_input_mode == "one_pic_T":
        style_reference_features1_T = layer_features[2, :, :, :]
        style_reference_features_mix = 0.5 * (style_reference_features1 + style_reference_features1_T)
        #style_reference_features_mix = K.maximum(style_reference_features1, style_reference_features1_T)
    else:  # image_input_mode == "two_pic"
        style_reference_features2 = layer_features[3, :, :, :]
        k = two_style_k
        style_reference_features_mix = style_reference_features1 * k + style_reference_features2 * (1-k)

    loss += w * style_loss(style_reference_features_mix, combination_features)


# 求得梯度，输入combination_image，对loss求梯度
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):  # 输入x，输出对应于x的梯度和loss
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, height, width))
    else:
        x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])  # 输入x，得到输出
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# Evaluator可以只需要进行一次计算就能得到所有的梯度和loss
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()
x = preprocess_image(char_image)
img = deprocess_image(x.copy())
fname = result_prefix + chars + '_原始图片.png'
save_img(fname, img)

# 开始迭代
for i in range(iterations):
    start_time = time.time()
    print('代数', i,end="   ")
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20, epsilon=1e-7)
    # 一个scipy的L-BFGS优化器
    print('目前loss:', min_val,end="  ")
    # 保存生成的图片
    img = deprocess_image(x.copy())
    fname = result_prefix + chars + '_代数_%d.png' % i
    end_time = time.time()
    print('耗时%.2f s' % (end_time - start_time))

    if i%5 == 0 or i == iterations-1:
        save_img(fname, img, image_enhance=image_enhance)
        print('文件保存为', fname)
