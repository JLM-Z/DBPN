import tensorlayer as tl
from tensorflow_core.python.keras.layers import Subtract
from tensorlayer.layers import *
import pickle

def up_sampling(inputs, shape, w_init, b_init, gamma_init, scale, block, is_train=False, reuse = False):
    """
    Parameters
    ----------
    inputs 输入低分辨率数据
    shape 输入低分辨率图片的shape
    scale 放大的尺寸
    block 表示第几个上采样模块
    is_train 是否训练
    reuse 是否重用变量
    Returns out 经过上采样后的特征图
    -------
    """
    # w_init = tf.truncated_normal_initializer(stddev=0.01)
    # b_init = tf.constant_initializer(value=0.0)
    # gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope('up_projection_' + str(block)):
        tl.layers.set_name_reuse(reuse)
        # inputs = InputLayer(x, name="input")  # 将输入数据转化为
        # _, w, h, c = x.shape()   # 获取输入数据的各个维度
        up1 = DeConv2d(inputs, 32, (4, 4), (shape[0] * scale, shape[1] * scale), padding="SAME",  # 上采样层 输出维度为（None,
                       W_init=w_init, b_init=b_init, name="deconv1")    #  w * scale, h * scale, 64）
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name="bn1")  # 标准化

        conv1 = Conv2d(up1, 32, (4, 4), (scale, scale), act=None, padding="SAME",   # 下采样层  输出的维度与inputs的维度相同
                       W_init=w_init, b_init=b_init, name="conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2),       # 标准化  激活函数为lrelu
                               is_train=is_train, gamma_init=gamma_init, name="bn2")

        err_ = ElementwiseLayer([conv1, inputs], tf.subtract, name="error")     # 计算误差

        up2 = DeConv2d(err_, 32, (4, 4), (shape[0] * scale, shape[1] * scale), padding="SAME",    # 将上一步的误差转化为相应的高分辨的补充值
                       W_init=w_init, b_init=b_init, name="deconv2")
        up2 = BatchNormLayer(up2, act=tf.nn.tanh, is_train=is_train, gamma_init=gamma_init, name="bn3")  # 标准化 使用tanh作为激活函数

        out = ElementwiseLayer([up1, up2], tf.add, name="up_out")   # 将误差与上采样后的特征图进行相加
        out.outputs = tl.act.ramp(out.outputs, v_min=-1, v_max=1)   # 限制范围

        return out

def down_sampling(inputs, shape, w_init, b_init, gamma_init, scale, block, is_train=False, reuse = False):
    """
    Parameters
    ----------
    inputs  输入的高分辨率图片
    shape 输入高分辨率图片的shape
    w_init 权重初始化
    b_init 偏置初始化
    gamma_init 标准化的参数
    scale  需缩小的尺寸
    block  第几个下采样块
    is_train  是否训练
    reuse  是否重用变量

    Returns  out  下采样后的图片
    -------

    """
    # w_init = tf.truncated_normal_initializer(stddev=0.01)
    # b_init = tf.constant_initializer(value=0.0)
    # gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope('down_projection_' + str(block)):
        tl.layers.set_name_reuse(reuse)
        # inputs = InputLayer(x, name="input")  # 将输入数据转化为
        # _, w, h, c = x.shape()  # 获取输入数据的各个维度
        conv1 = Conv2d(inputs, 32, (4, 4), (scale, scale), act=None, padding="SAME",    # 下采样层
                       W_init=w_init, b_init=b_init, name="conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2),       # 标准化  激活函数为lrelu
                               is_train=is_train, gamma_init=gamma_init, name="bn1")

        up1 = DeConv2d(conv1, 32, (4, 4), out_size=(shape[0], shape[1]), padding="SAME",  # 上采样层  输出的特征图维度应和输入数据的相同
                       W_init=w_init, b_init=b_init, name="deconv1")
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name="bn2")  # 标准化

        err_ = ElementwiseLayer([up1, inputs], tf.subtract, name="error")  # 计算误差
        # err_ = Subtract()[up1.outputs, inputs.outputs]  # 误差

        conv2 = Conv2d(err_, 32, (4, 4), (scale, scale), act=None, padding="SAME",    # 下采样层
                       W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=tf.nn.tanh, is_train=is_train, gamma_init=gamma_init, name="bn3")

        out = ElementwiseLayer([conv1, conv2], tf.add, name="down_out")  # 将误差与上采样后的特征图进行相加
        out.outputs = tl.act.ramp(out.outputs, v_min=-1, v_max=1)  # 限制范围
        return out

def back_projection(x, scale, is_train, reuse,n_projeciton=2):
    """

    Parameters
    ----------
    x 输入的低分辨率图片
    scale 需方法的尺寸
    reuse 是否重用参数
    n_projeciton up- and down-sampling 共迭代几次

    Returns  out 网络计算的最终结果
    -------

    """
    # ======================= 初始化 ==================
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    # ======================= define DBPN ==================
    with tf.variable_scope("back_projection"):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name="input")  # 处理输入数据的格式
        _, w, h, c = x.shape  # 获取输入图像的各个维度
        up_list = []
        x = Conv2d(inputs, 64, filter_size=(3, 3), strides=(1, 1), padding="SAME",
                   W_init=w_init, b_init=b_init, name="conv1")
        x = Conv2d(x, 32, filter_size=(1, 1), strides=(1, 1), padding="SAME",
                   W_init=w_init, b_init=b_init, name="conv2")
        for i in range(n_projeciton):
            x = up_sampling(inputs=x, shape=[w, h], w_init=w_init, b_init=b_init, gamma_init=gamma_init,
                            scale=scale, block=i+1, is_train=is_train, reuse=reuse)  # 上采样
            up_list.append(x)  # 保存上采样的数据
            x = down_sampling(inputs=x, shape=[w*scale, h*scale], w_init=w_init, b_init=b_init, gamma_init=gamma_init,
                              scale=scale, block=i+1, is_train=is_train, reuse=reuse)  # 下采样
        x = up_sampling(x, shape=[w, h], w_init=w_init, b_init=b_init, gamma_init=gamma_init,
                        scale=scale, block=n_projeciton+1, is_train=is_train, reuse=reuse) # 上采样
        i = 0
        for up_img in up_list:      # 连接前面所有的上采样的特征图
            i = i + 1
            x = ConcatLayer([x, up_img], concat_dim=3,name='concat_'+str(i))

        out = Conv2d(x, n_filter=1, filter_size=(3, 3), act=tf.nn.tanh,padding="SAME", name="BP_output")
        out.outputs = tl.act.ramp(out.outputs, v_min=-1, v_max=1)  # 限制范围

        return out

if __name__=="__main__":
    with open("images\\testing.pickle", 'rb') as f:
        X_test = pickle.load(f)
    ind = np.random.randint(low=0, high=X_test.shape[0]-1, size=50)
    Test = X_test[ind]
    _,nw,nh,nz= X_test.shape
    t_image_good = tf.placeholder('float32', [50, nw, nh, nz], name='good_image')

    net = back_projection(t_image_good,scale=2,is_train=True,reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(net,feed_dict={t_image_good:Test})

