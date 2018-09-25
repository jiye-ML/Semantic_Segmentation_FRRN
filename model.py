import numpy as np
from commons.ops import *

# 下面是FRRU网络架构，根据原来论文中table1的a架构实现
def _arch_type_a(num_classes):
    # 残差连接： 两个 3x3 + bn + relu
    def _ru(t, conv3_1, bn_1, conv3_2, bn_2):
        _t = tf.nn.relu(bn_1(conv3_1(t)))
        _t = conv3_2(_t)
        _t = bn_2(_t)
        return t + _t

    # FRRU 操作，论文的核心模块， 论文中有结构的示意图
    def _frru(y_z, conv3_1, bn_1, conv3_2, bn_2, conv1, scale):
        # 输入由两个组成 上面线路的 z 和下面线路的 y
        y, z = y_z
        # 将上面的池化后与下面的输入连接
        _t = tf.concat([y, tf.nn.max_pool(z, [1, scale, scale, 1], [1, scale, scale, 1], 'SAME')], axis=3)
        # 第一个卷积模块
        _t = tf.nn.relu(bn_1(conv3_1(_t)))
        # 第二个卷积模块
        y_prime = tf.nn.relu(bn_2(conv3_2(_t)))
        # 降维
        _t = conv1(y_prime)
        # 上采样到z一样大小，便于残差连接
        _t = tf.image.resize_nearest_neighbor(_t, tf.shape(y_prime)[1 : 3] * scale)
        # 残差连接
        z_prime = _t + z
        # 输出也是两部分，上面部分z负责学习全分辨率，下面y负责学习池化后的特征
        return y_prime, z_prime

    # 分流: 将原来的输入分成上下两条流
    def _divide_stream(t, conv1):
        return t, conv1(t)

    # 融合两条流，
    def _concat_stream(y_z, conv1):
        # 对y上采样然后与z融合
        return conv1(tf.concat([tf.image.resize_bilinear(y_z[0], tf.shape(y_z[0])[1 : 3] * 2), y_z[1]], axis=3))

    '''
    下面是FRRU网络架构，根据原来论文中table1实现
    '''
    # functools.partial 通过包装手法，允许我们 "重新定义" 函数签名用一些默认参数包装一个可调用对象,
    # 返回结果是可调用对象，并且可以像原始对象一样对待冻结部分函数位置函数或关键字参数，简化函数,更少更灵活的函数参数调用
    from functools import partial
    # 第一个卷积模块
    spec = [Conv2d('conv2d_1', 3, 48, 5, 5, 1, 1, data_format='NHWC'),
            BatchNorm('conv2d_1_bn', 48, axis=3), lambda t, **kwargs : tf.nn.relu(t)]
    # RU Layers
    for i in range(3):
        spec.append(
            partial(_ru,
                    conv3_1 = Conv2d('ru48_{}_1'.format(i), 48, 48, 3, 3, 1, 1, data_format='NHWC'),
                    bn_1 = BatchNorm('ru48_{}_1_bn'.format(i), 48, axis=3),
                    conv3_2 = Conv2d('ru48_{}_2'.format(i), 48, 48, 3, 3, 1, 1, data_format='NHWC'),
                    bn_2 = BatchNorm('ru48_{}_2_bn'.format(i), 48, axis=3))
        )
    # 将流分成两部分,上下两条线，一条学习全分辨率，一条学习降采样后的特征
    spec.append(partial(_divide_stream, conv1 = Conv2d('conv32', 48, 32, 1, 1, 1, 1,data_format='NHWC')))

    # FFRU Layers (编码)
    # 输入的通道
    prev_ch = 48
    # （这一个模块中FRRU的个数，输出通道，分辨率缩小的尺度）
    for it, ch, scale in [(3, 96, 2), (4, 192, 4),(2, 384, 8),(2, 384, 16)] :
        # 对y降采样， 每个FRRus模块开始前先对y进行下采样
        spec.append(lambda y_z : (tf.nn.max_pool(y_z[0], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'), y_z[1]))
        for i in range(it):
            # 这里+32是因为残差流的卷积模块总是32通道的，所以两个特征融合后多了32
            spec.append(
                partial(_frru,
                        conv3_1 = Conv2d('encode_frru{}_{}_{}_1'.format(ch, scale, i), prev_ch + 32, ch, 3, 3, 1, 1, data_format='NHWC'),
                        bn_1 = BatchNorm('encode_frru{}_{}_{}_1_bn'.format(ch, scale, i), ch, axis=3),
                        conv3_2 = Conv2d('encode_frru{}_{}_{}_2'.format(ch, scale, i), ch, ch, 3, 3, 1, 1, data_format='NHWC'),
                        bn_2 = BatchNorm('encode_frru{}_{}_{}_2_bn'.format(ch, scale, i), ch, axis=3),
                        conv1 = Conv2d('encode_frru{}_{}_{}_3'.format(ch, scale, i), ch, 32, 1, 1, 1, 1, data_format='NHWC'),
                        scale = scale)
            )
            prev_ch = ch
            pass
        pass

    # FRRU Layers (解码)
    for it, ch, scale in [(2, 192, 8), (2, 192, 4), (2, 96, 2)]:
        # 解码的时候先对y进行上采样
        spec.append(lambda y_z : (tf.image.resize_bilinear(y_z[0], tf.shape(y_z[0])[1:3]*2), y_z[1]))
        for i in range(it):
            spec.append(
                partial(_frru,
                        conv3_1=Conv2d('decode_frru{}_{}_{}_1'.format(ch, scale, i),prev_ch+32,ch,3,3,1,1,data_format='NHWC'),
                        bn_1 = BatchNorm('decode_frru{}_{}_{}_1_bn'.format(ch, scale, i),ch,axis=3),
                        conv3_2=Conv2d('decode_frru{}_{}_{}_2'.format(ch, scale, i),ch,ch,3,3,1,1,data_format='NHWC'),
                        bn_2 = BatchNorm('decode_frru{}_{}_{}_2_bn'.format(ch, scale, i),ch,axis=3),
                        conv1 = Conv2d('decode_frru{}_{}_{}_3'.format(ch, scale, i),ch,32,1,1,1,1,data_format='NHWC'),
                        scale=scale)
            )
            prev_ch = ch
            pass
        pass

    # 融合流
    spec.append(partial(_concat_stream, conv1 = Conv2d('conv48',prev_ch + 32, 48, 1, 1, 1, 1, data_format='NHWC')))

    # 残差模块
    for i in range(3,6):
        spec.append(
            partial(_ru,
                    conv3_1=Conv2d('ru48_{}_1'.format(i), 48, 48, 3, 3, 1, 1, data_format='NHWC'),
                    bn_1 = BatchNorm('ru48_{}_1_bn'.format(i),48, axis = 3),
                    conv3_2=Conv2d('ru48_{}_2'.format(i), 48, 48, 3, 3, 1, 1, data_format = 'NHWC'),
                    bn_2 = BatchNorm('ru48_{}_2_bn'.format(i),48,axis=3))
        )

    # 分类层
    spec.append(Conv2d('conv_c', 48, num_classes, 1, 1, 1, 1, data_format='NHWC'))

    return spec

class FRRN():

    def __init__(self, lr, global_step, K, im, gt, arch_fn, param_scope, is_training=False):
        # 网络结构
        with tf.variable_scope(param_scope):
             net_spec = arch_fn()

        self.logits = None
        self.preds = None
        self.train_op = None
        with tf.variable_scope('forward') as forward_scope:
            _t = im
            for block in net_spec:
                print(_t)
                _t = block(_t)
            self.logits = _t
            self.preds = tf.argmax(self.logits, axis=3)

            # Loss
            naive_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = gt)
            #  ignore pixels labed as void? is it requried?
            mask = tf.cast(tf.logical_not(tf.equal(gt, 0)), tf.float32)
            naive_loss = naive_loss * mask
            # boostrap loss 论文中的损失函数，
            boot_loss, _ = tf.nn.top_k(tf.reshape(naive_loss, [tf.shape(im)[0], tf.shape(im)[1] * tf.shape(im)[2]]), k = K, sorted = False)
            self.loss = tf.reduce_mean(tf.reduce_sum(boot_loss, axis=1))
            pass

        if (is_training):
            with tf.variable_scope('backward'):
                optimizer = tf.train.AdamOptimizer(lr)

                update_ops = tf.get_collection(tf.GraphKeys. UPDATE_OPS, forward_scope.name)
                #print('------------batchnorm ops---------------')
                #for op in update_ops:
                #    print(update_ops)
                #print('------------batchnorm ops end---------------')
                with tf.control_dependencies(update_ops):
                    self.train_op= optimizer.minimize(self.loss, global_step = global_step)

        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        #for name,var in save_vars.items():
        #    print(name,var)

        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)
        pass

    # 模型保存
    def save(self, sess, dir, step=None):
        if(step is not None):
            self.saver.save(sess,dir + '/model.ckpt', global_step=step)
        else :
            self.saver.save(sess,dir + '/last.ckpt')
        pass

    # 模型加载
    def load(self, sess, model):
        self.saver.restore(sess, model)
        pass

    pass


if __name__ == "__main__":

    with tf.variable_scope('params') as params:
        pass

    im = tf.placeholder(tf.float32,[None, 256, 512, 3])
    gt = tf.placeholder(tf.int32,[None, 256, 512]) #19 + unlabeled area(plus ignored labels)
    global_step = tf.Variable(0, trainable=False)

    from functools import partial
    net = FRRN(0.1, global_step, 512*64, im, gt, partial(_arch_type_a, 20), params, True)
    print(net.logits)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    for _ in range(30):
        _t, preds, _ = (sess.run([net. logits, net.preds, net.train_op],
                               feed_dict={im: np.random.random((1,256, 512, 3)),
                                          gt: np.zeros((1, 256, 512))}))
        print(preds.shape)

