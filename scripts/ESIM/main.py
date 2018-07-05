import numpy
import copy
import sys
import logging
import pprint
from collections import OrderedDict
from data_iterator import TextIterator
from data_iterator import prepare_data
import os
import pickle as pkl
import time
import tensorflow as tf
import pandas as pd
# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')
def _p(pp, name):
    return '%s_%s' % (pp, name)

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def RNN_layer(state_below,mask,options,params,prefix):
    state_below=tf.transpose(state_below,[1,0,2])
    mask = tf.transpose(mask, [1, 0])
    n_samples = tf.shape(state_below)[1]
    dim=int(params[_p(prefix,'U')].shape[0])
    init_state = tf.zeros([n_samples,dim])
    init_memory = tf.zeros([n_samples,dim])
    input_gate = tf.zeros([n_samples,dim])
    output_gate = tf.zeros([n_samples,dim])
    forget_gate = tf.zeros([n_samples,dim])
    # use the slice to calculate all the different gates
    def _slice(_x, n, dim):
        if len(_x.shape) == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        elif len(_x.shape) == 2:
            return _x[:, n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]

    # one time step of the lstm
    def _step(a,b):
        h_,c_=a[:2]
        m_, x_=b

        preact = tf.matmul(h_, params[_p(prefix, 'U')])
        preact += x_

        i = tf.nn.sigmoid(_slice(preact, 0, dim))
        f = tf.nn.sigmoid(_slice(preact, 1, dim))
        o = tf.nn.sigmoid(_slice(preact, 2, dim))
        c = tf.tanh(_slice(preact, 3, dim))
        c = f * c_ + i * c

        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tf.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return [h, c, i, f, o]

    s=tf.shape(state_below)
    state_below = tf.matmul(tf.reshape(state_below,[-1,tf.shape(state_below)[-1]]), params[_p(prefix, 'W')]) + params[_p(prefix, 'b')]
    state_below=tf.reshape(state_below,tf.concat([s[:2],[-1]],0))

    rval=tf.scan(_step,
                 [mask,state_below],
                 [init_state,init_memory,input_gate,forget_gate,output_gate])
    return rval

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim']
    if nout is None:
        nout = options['dim']
    params[_p(prefix, 'W')] = tf.Variable(norm_weight(nin, nout, scale=0.01, ortho=ortho))
    params[_p(prefix, 'b')] = tf.Variable(numpy.zeros((nout,)).astype('float32'))

    return params






# input variable
use_noise=tf.placeholder(tf.bool)
word_x1=tf.placeholder(tf.int32,[None,None])
word_x1_mask=tf.placeholder(tf.float32,[None,None])
word_x2=tf.placeholder(tf.int32,[None,None])
word_x2_mask = tf.placeholder(tf.float32, [None,None])
char_x1_mask=tf.placeholder(tf.float32, [None,None,None])
char_x2_mask=tf.placeholder(tf.float32,[None,None,None])
char_x1=tf.placeholder(tf.int32, [None,None,None])
char_x2=tf.placeholder(tf.int32,[None,None,None])
y=tf.placeholder(tf.int32,[None])

def build_model(options,worddicts):
    """
    Builds the entire computational graph used for trainning
    :param model_options:
    :return: opt_ret,cost,f_pred,r_prods
    """
    opt_ret=dict()
    params=dict()
    word_xr1_mask=tf.reverse(word_x1_mask,[1])
    word_xr2_mask = tf.reverse(word_x2_mask, [1])



    #embedding layer
    word_embedding = norm_weight(options['n_words'], options['dim_word'])
    if options['embedding']:
        with open(options['embedding'], 'r',encoding='iso-8859-1') as f:
                for line in f:
                    temp=line.split()
                    word=temp[0]
                    vector=temp[1:]
                    if word in worddicts and worddicts[word]<options['n_words']:
                        word_embedding[worddicts[word],:]=vector

    word_embedding_layer=tf.Variable(word_embedding,name='word_embedding')

    emb1=tf.nn.embedding_lookup(word_embedding_layer,word_x1,name='embedding_word_lookup1')
    emb2=tf.nn.embedding_lookup(word_embedding_layer,word_x2,name='embedding_word_lookup2')

    if options['use_dropout']:
        emb1=tf.cond(use_noise,lambda :tf.nn.dropout(emb1,0.5),lambda :emb1)
        emb2 = tf.cond(use_noise, lambda: tf.nn.dropout(emb2, 0.5), lambda: emb2)

    #1-layer LSTM
    print('LSTM result')
    for l in range(1):
        #param_init_lstm
        prefix = 'encoder_{}'.format(str(l + 1))
        if l==0:
            nin=options['dim_word']
        else:
            nin = options['dim_word']+2*options['dim']
        dim=options['dim']

        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = tf.Variable(W)

        # for the previous hidden activation
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = tf.Variable(U)
        params[_p(prefix, 'b')] = tf.Variable(numpy.zeros((4 * dim,)).astype('float32'))

        #param_init_rlstm
        prefix = 'encoder_r_{}'.format(str(l + 1))
        if l==0:
            nin=options['dim_word']
        else:
            nin = options['dim_word'] +2*options['dim']
        dim=options['dim']

        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = tf.Variable(W)

        # for the previous hidden activation
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = tf.Variable(U)
        params[_p(prefix, 'b')] = tf.Variable(numpy.zeros((4 * dim,)).astype('float32'))



        if l==0:
            ctx1=emb1
            ctx2=emb2
        else:
            ctx1=tf.concat([ctx1,emb1],axis=2)
            ctx2=tf.concat([ctx2,emb2],axis=2)

        print(ctx1)

        ctxr1=tf.reverse(ctx1,axis=[1])
        ctxr2=tf.reverse(ctx2,axis=[1])

        proj1=RNN_layer(ctx1,word_x1_mask,options,params,prefix='encoder_{}'.format(str(l+1)))
        projr1=RNN_layer(ctxr1,word_xr1_mask,options,params,prefix='encoder_r_{}'.format(str(l+1)))
        proj2=RNN_layer(ctx2,word_x2_mask,options,params,prefix='encoder_{}'.format(str(l+1)))
        projr2=RNN_layer(ctxr2,word_xr2_mask,options,params,prefix='encoder_r_{}'.format(str(l+1)))

        ctx1=tf.concat([proj1[0],projr1[0][::-1]],axis=len(projr1[0].shape)-1)
        ctx2 = tf.concat([proj2[0], projr2[0][::-1]], axis=len(projr2[0].shape) - 1)
        ctx1 = tf.transpose(ctx1, [1, 0, 2])
        ctx2 = tf.transpose(ctx2, [1, 0, 2])
        print(ctx1)

    ctx1=ctx1*word_x1_mask[:,:,None]
    ctx2 = ctx2 * word_x2_mask[:, :, None]
    def _step(h,x):
        return tf.matmul(x[0],x[1])
    temp=tf.zeros((tf.shape(ctx1)[1],tf.shape(ctx2)[1]))
    weight_martrix=tf.scan(_step,[ctx1,tf.transpose(ctx2,[0,2,1])],temp)
    weight_martrix_1=tf.exp(weight_martrix)*word_x2_mask[:,None,:]
    weight_martrix_2=tf.transpose(tf.exp(weight_martrix)*word_x1_mask[:,:,None],[0,2,1])
    weight_martrix_1=weight_martrix_1/tf.reduce_sum(weight_martrix_1,axis=2)[:,:,None]
    weight_martrix_2 = weight_martrix_2 / tf.reduce_sum(weight_martrix_2, axis=2)[:,:,None]

    ctx1_=tf.reduce_sum(weight_martrix_1[:,:,:,None]*ctx2[:,None,:,:],axis=2)
    ctx2_ = tf.reduce_sum(weight_martrix_2[:, :, :, None] * ctx1[:, None, :, :],axis=2)
    inp1=tf.concat([ctx1, ctx1_, ctx1*ctx1_, ctx1-ctx1_],axis=2)
    inp2 = tf.concat([ctx2, ctx2_, ctx2 * ctx2_, ctx2 - ctx2_], axis=2)
    params = param_init_fflayer(options, params, prefix='projection',
                                nin=options['dim'] * 8, nout=options['dim'], ortho=False)


    s=tf.shape(inp1)
    inp1 = tf.nn.relu(tf.matmul(tf.reshape(inp1,[-1,int(inp1.shape[-1])]), params[_p('projection', 'W')]) + params[_p('projection', 'b')])
    inp1=tf.reshape(inp1,tf.concat([s[:2],[-1]],0))
    s=tf.shape(inp2)
    inp2 = tf.nn.relu(tf.matmul(tf.reshape(inp2,[-1,int(inp2.shape[-1])]), params[_p('projection', 'W')]) + params[_p('projection', 'b')])
    inp2=tf.reshape(inp2,tf.concat([s[:2],[-1]],0))
    if options['use_dropout']:
        inp1=tf.cond(use_noise,lambda :tf.nn.dropout(inp1,0.5),lambda :inp1)
        inp2 = tf.cond(use_noise, lambda: tf.nn.dropout(inp2, 0.5), lambda: inp2)


    for l in range(1):
        #param_init_lstm
        prefix = 'decoder_{}'.format(str(l + 1))
        if l==0:
            nin=options['dim']
        else:
            nin = options['dim']+2*options['dim']
        dim=options['dim']

        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = tf.Variable(W)

        # for the previous hidden activation
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = tf.Variable(U)
        params[_p(prefix, 'b')] = tf.Variable(numpy.zeros((4 * dim,)).astype('float32'))

        #param_init_rlstm
        prefix = 'decoder_r_{}'.format(str(l + 1))
        if l==0:
            nin=options['dim']
        else:
            nin = options['dim'] +2*options['dim']
        dim=options['dim']

        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = tf.Variable(W)

        # for the previous hidden activation
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = tf.Variable(U)
        params[_p(prefix, 'b')] = tf.Variable(numpy.zeros((4 * dim,)).astype('float32'))



        if l==0:
            ctx1=inp1
            ctx2=inp2
        else:
            ctx1=tf.concat([ctx1,inp1],axis=2)
            ctx2=tf.concat([ctx2,inp2],axis=2)

        print(ctx1)

        ctxr1=tf.reverse(ctx1,axis=[1])
        ctxr2=tf.reverse(ctx2,axis=[1])

        proj1=RNN_layer(ctx1,word_x1_mask,options,params,prefix='decoder_{}'.format(str(l+1)))
        projr1=RNN_layer(ctxr1,word_xr1_mask,options,params,prefix='decoder_r_{}'.format(str(l+1)))
        proj2=RNN_layer(ctx2,word_x2_mask,options,params,prefix='decoder_{}'.format(str(l+1)))
        projr2=RNN_layer(ctxr2,word_xr2_mask,options,params,prefix='decoder_r_{}'.format(str(l+1)))

        ctx1=tf.concat([proj1[0],projr1[0][::-1]],axis=len(projr1[0].shape)-1)
        ctx2 = tf.concat([proj2[0], projr2[0][::-1]], axis=len(projr2[0].shape) - 1)
        ctx1 = tf.transpose(ctx1, [1, 0, 2])
        ctx2 = tf.transpose(ctx2, [1, 0, 2])
        print(ctx1)

    mean_1=tf.reduce_sum(ctx1*word_x1_mask[:,:,None],axis=1)/tf.reduce_sum(word_x1_mask,axis=1)[:,None]
    max_1=tf.reduce_max(ctx1*word_x1_mask[:,:,None],axis=1)

    mean_2=tf.reduce_sum(ctx2*word_x2_mask[:,:,None],axis=1)/tf.reduce_sum(word_x2_mask,axis=1)[:,None]
    max_2=tf.reduce_max(ctx2*word_x2_mask[:,:,None],axis=1)

    #represention and MLP layer
    logit=tf.concat([mean_1,mean_2,max_1,max_2],axis=1)
    if options['use_dropout']:
        logit=tf.cond(use_noise,lambda :tf.nn.dropout(logit,0.5),lambda :logit)


    params = param_init_fflayer(options, params, prefix='ff_layer_1',
                                nin=options['dim'] * 8, nout=options['dim'], ortho=False)
    params = param_init_fflayer(options, params, prefix='ff_layer_output',
                                nin=options['dim'], nout=3, ortho=False)
    logit=tf.nn.tanh(tf.matmul(logit,params[_p('ff_layer_1','W')])+params[_p('ff_layer_1','b')])
    if options['use_dropout']:
        logit=tf.cond(use_noise,lambda :tf.nn.dropout(logit,0.5),lambda :logit)

    logit=tf.matmul(logit, params[_p('ff_layer_output', 'W')]) + params[_p('ff_layer_output', 'b')]
    probs=tf.nn.softmax(logit)
    pred=tf.argmax(probs,1)
    cost=tf.losses.sparse_softmax_cross_entropy(y,logit)
    return opt_ret,cost,pred,probs




logger = logging.getLogger(__name__)
def train(
          dim_word         = 100,  # word vector dimensionality
          dim              = 100,  # the number of GRU units
          encoder          = 'lstm', # encoder model
          decoder          = 'lstm', # decoder model
          patience         = 10,  # early stopping patience
          max_epochs       = 5000,
          finish_after     = 10000000, # finish after this many updates
          decay_c          = 0.,  # L2 regularization penalty
          clip_c           = -1.,  # gradient clipping threshold
          lrate            = 0.01,  # learning rate
          n_words          = 100000,  # vocabulary size
          maxlen           = 100,  # maximum length of the description
          optimizer        = 'adadelta',
          batch_size       = 16,
          valid_batch_size = 16,
          saveto           = 'model.npz',
          LoadFrom         = '',
          dispFreq         = 100,
          validFreq        = 1000,
          saveFreq         = 1000,   # save the parameters after every saveFreq updates
          use_dropout      = False,
          reload_          = False,
          test          = 1, # print verbose information for debug but slow speed
          datasets         = [],
          valid_datasets   = [],
          test_datasets    = [],
          test_matched_datasets=[],
          test_mismatched_datasets=[],
          dictionary       = '',
          embedding        = '', # pretrain embedding file, such as word2vec, GLOVE
          ):
    logging.basicConfig(level=logging.DEBUG,format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    #model_options
    model_options=locals().copy()
    model_options['alphabet'] = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    model_options['l_alphabet'] = len(model_options['alphabet'])
    model_options['dim_char_emb'] = 15
    model_options['char_nout'] = 100
    model_options['char_k_rows'] = 5
    model_options['char_k_cols'] = model_options['dim_char_emb']

    #load dictionary and invert them
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f,encoding='iso-8859-1')
    worddicts_r=dict()
    for word in worddicts:
        worddicts_r[worddicts[word]]=word

    logger.debug(pprint.pformat(model_options))

    time.sleep(0.1)
    print('Loading data')

    #return (3,batch_size,-1)
    train = TextIterator(datasets[0], datasets[1], datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=batch_size)
    train_valid = TextIterator(datasets[0], datasets[1], datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    valid = TextIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    test = TextIterator(test_datasets[0], test_datasets[1], test_datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    test_matched = TextIterator(test_matched_datasets[0], test_matched_datasets[1], test_matched_datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    test_mismatched = TextIterator(test_mismatched_datasets[0], test_mismatched_datasets[1], test_mismatched_datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    print('Building model')
    opt_ret,cost,pred,probs=build_model(model_options,worddicts)
    op=tf.train.AdamOptimizer(model_options['lrate'],beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cost)

    uidx=0
    eidx=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if model_options['reload_']:
            saver = tf.train.Saver()
            saver.restore(sess,model_options['LoadFrom'])
            print('Reload dond!')
        train_loss=0
        while True:
            try:
                x1,x2,label=train.next()
            except:
                eidx+=1
                print(eidx)
                continue
            _x1, _x1_mask, _char_x1, _char_x1_mask, _x2, _x2_mask, _char_x2, _char_x2_mask,lengths_x,lengths_y, _y=prepare_data(x1, x2, label, worddicts_r,model_options['alphabet'], maxlen=maxlen)
            ud_start = time.time()
            _cost,_pred,_prob,_=sess.run([cost,pred,probs,op],feed_dict={use_noise:True,
                                     word_x1:_x1,word_x1_mask:_x1_mask,char_x1:_char_x1,
                                     word_x2: _x2, word_x2_mask: _x2_mask, char_x2: _char_x2,
                                     char_x1_mask:_char_x1_mask,char_x2_mask:_char_x2_mask,
                                     y:_y
                                     })
            ud = time.time() - ud_start
            uidx+=1
            train_loss+=_cost
            if uidx%model_options['dispFreq']==0:
                logger.debug('Epoch {0} Update {1} Cost {2} UD {3}'.format(eidx, uidx, train_loss/model_options['dispFreq'], ud,))
                train_loss=0
            if uidx % model_options['validFreq'] == 0:
                valid_cost=0
                valid_pred=[]
                valid_label=[]
                n_vaild_samples=0
                test_cost=0
                test_pred=[]
                test_label=[]
                n_test_samples = 0
                while True:
                    try:
                        x1, x2, label = valid.next()
                        _x1, _x1_mask, _char_x1, _char_x1_mask, _x2, _x2_mask, _char_x2, _char_x2_mask, lengths_x, lengths_y, _y = prepare_data(
                            x1, x2, label, worddicts_r, model_options['alphabet'], maxlen=maxlen)
                        _cost, _pred, _prob = sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _x1,
                                                                                              word_x1_mask: _x1_mask,
                                                                                              char_x1: _char_x1,
                                                                                              word_x2: _x2,
                                                                                              word_x2_mask: _x2_mask,
                                                                                              char_x2: _char_x2,
                                                                                              char_x1_mask: _char_x1_mask,
                                                                                              char_x2_mask: _char_x2_mask,
                                                                                              y: _y
                                                                                              })
                        valid_cost+=_cost*len(label)
                        valid_pred.extend(_pred)
                        valid_label.extend(_y)
                        n_vaild_samples+=len(label)
                        print('Seen %d samples' % n_vaild_samples)
                    except:
                        break

                while True:
                    try:
                        x1, x2, label = test.next()
                        _x1, _x1_mask, _char_x1, _char_x1_mask, _x2, _x2_mask, _char_x2, _char_x2_mask, lengths_x, lengths_y, _y = prepare_data(
                            x1, x2, label, worddicts_r, model_options['alphabet'], maxlen=maxlen)
                        _cost, _pred, _prob = sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _x1,
                                                                                              word_x1_mask: _x1_mask,
                                                                                              char_x1: _char_x1,
                                                                                              word_x2: _x2,
                                                                                              word_x2_mask: _x2_mask,
                                                                                              char_x2: _char_x2,
                                                                                              char_x1_mask: _char_x1_mask,
                                                                                              char_x2_mask: _char_x2_mask,
                                                                                              y: _y
                                                                                              })
                        test_cost+=_cost*len(label)
                        test_pred.extend(_pred)
                        test_label.extend(_y)
                        n_test_samples+=len(label)
                        print('Seen %d samples' % n_test_samples)
                    except:
                        print('Valid cost',valid_cost/len(valid_label))
                        print('Valid accuracy',numpy.mean(numpy.array(valid_pred)==numpy.array(valid_label)))
                        print('Test cost',test_cost/len(test_label))
                        print('Test accuracy',numpy.mean(numpy.array(test_pred)==numpy.array(test_label)))
                        break
            if uidx % model_options['test'] == 0:
                mismatched_result=[]
                matched_result=[]
                while True:
                    try:
                        x1, x2, label = test_mismatched.next()
                        _x1, _x1_mask, _char_x1, _char_x1_mask, _x2, _x2_mask, _char_x2, _char_x2_mask, lengths_x, lengths_y, _y = prepare_data(
                            x1, x2, label, worddicts_r, model_options['alphabet'], maxlen=maxlen)
                        _cost, _pred, _prob = sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _x1,
                                                                                              word_x1_mask: _x1_mask,
                                                                                              char_x1: _char_x1,
                                                                                              word_x2: _x2,
                                                                                              word_x2_mask: _x2_mask,
                                                                                              char_x2: _char_x2,
                                                                                              char_x1_mask: _char_x1_mask,
                                                                                              char_x2_mask: _char_x2_mask,
                                                                                              y: _y
                                                                                              })
                        mismatched_result.extend(_pred)
                        print(len(mismatched_result))
                    except:
                        break
                while True:
                    try:
                        x1, x2, label = test_matched.next()
                        _x1, _x1_mask, _char_x1, _char_x1_mask, _x2, _x2_mask, _char_x2, _char_x2_mask, lengths_x, lengths_y, _y = prepare_data(
                            x1, x2, label, worddicts_r, model_options['alphabet'], maxlen=maxlen)
                        _cost, _pred, _prob = sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _x1,
                                                                                              word_x1_mask: _x1_mask,
                                                                                              char_x1: _char_x1,
                                                                                              word_x2: _x2,
                                                                                              word_x2_mask: _x2_mask,
                                                                                              char_x2: _char_x2,
                                                                                              char_x1_mask: _char_x1_mask,
                                                                                              char_x2_mask: _char_x2_mask,
                                                                                              y: _y
                                                                                              })
                        matched_result.extend(_pred)
                        print(len(matched_result))
                    except:
                        break
                index=0
                a=[]
                b=[]
                dic = {0:'entailment', 1: 'neutral', 2: 'contradiction'}
                for i in mismatched_result:
                    a.append((index,dic[i]))
                    index+=1
                for i in matched_result:
                    b.append((index,dic[i]))
                    index+=1
                a=pd.DataFrame(a)
                a.columns=['pairID','gold_label']
                a.to_csv('sub_mismatched_'+str(uidx)+'.csv',index=False)
                b=pd.DataFrame(b)
                b.columns=['pairID','gold_label']
                b.to_csv('sub_matched_'+str(uidx)+'.csv',index=False)
                print('submission '+str(uidx)+' done!')
            if uidx%model_options['saveFreq']==0:
                saver=tf.train.Saver()
                save_path=saver.save(sess,model_options['saveto']+'_'+str(uidx))
                print("Model saved in file: %s"%save_path)
