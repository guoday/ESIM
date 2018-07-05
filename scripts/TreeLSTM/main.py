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


def param_init_tree_lstm(options,params,prefix='tree_lstm',nin=None,dim=None):
    if nin is None:
        nin=options['dim']
    if dim is None:
        dim=options['dim']

    W=numpy.concatenate([norm_weight(nin,dim),
                         norm_weight(nin, dim),
                         norm_weight(nin, dim),
                         norm_weight(nin, dim)],axis=1)
    params[_p(prefix,'W')]=tf.Variable(W)
    U_l=numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)],axis=1)
    params[_p(prefix,'U_l')]=tf.Variable(U_l)
    U_r=numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)],axis=1)
    params[_p(prefix,'U_r')]=tf.Variable(U_r)
    params[_p(prefix, 'b')] = tf.Variable(numpy.zeros((4 * dim,)).astype('float32'))
    return params


def init_params(options,worddicts):
    params=dict()
    params['word_embedding'] = norm_weight(options['n_words'], options['dim_word'])
    if options['embedding']:
        with open(options['embedding'], 'r',encoding='iso-8859-1') as f:
                for line in f:
                    temp=line.split()
                    word=temp[0]
                    vector=temp[1:]
                    if word in worddicts and worddicts[word]<options['n_words']:
                        params['word_embedding'][worddicts[word],:]=vector
    params=param_init_tree_lstm(options,params,prefix='encoder',
                                nin=options['dim_word'],dim=options['dim'])
    params = param_init_tree_lstm(options, params, prefix='decoder',
                                nin=options['dim'], dim=options['dim'])
    params = param_init_fflayer(options, params, prefix='projection',
                                nin=options['dim'] *4, nout=options['dim'], ortho=False)
    params = param_init_fflayer(options, params, prefix='ff_layer_1',
                                nin=options['dim'] * 6, nout=options['dim'], ortho=False)
    params = param_init_fflayer(options, params, prefix='ff_layer_output',
                                nin=options['dim'], nout=3, ortho=False)
    return params

def tree_lstm_layer(params,input,options,prefix):
    state_below,mask,left_mask,right_mask=input
    n_samples = tf.shape(state_below)[1]
    n_steps = tf.shape(state_below)[0]
    dim=int(params[_p(prefix,'U_l')].shape[0])
    init_state = tf.zeros([n_samples,dim])
    init_memory = tf.zeros([n_samples,dim])
    h_ = tf.zeros([n_samples,n_steps,dim])
    c_ = tf.zeros([n_samples,n_steps,dim])
    # use the slice to calculate all the different gates
    def _slice(_x, n, dim):
        if len(_x.shape) == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        elif len(_x.shape) == 2:
            return _x[:, n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]
    def _step(a,b):
        nonlocal h_,c_
        m_, x_,left_mask,right_mask,counter_=b
        preact_l=tf.matmul(tf.reduce_sum(left_mask[:,:,None]*h_,axis=1),params[_p(prefix,'U_l')])
        preact_r = tf.matmul(tf.reduce_sum(right_mask[:, :, None] * h_, axis=1), params[_p(prefix, 'U_r')])
        x_=tf.concat([_slice(x_,0,dim),
                      _slice(x_,1,dim),
                      _slice(x_,1,dim),
                      _slice(x_,2,dim),
                      _slice(x_,3,dim)],axis=1)

        preact=preact_l+preact_r+x_
        i=tf.sigmoid(_slice(preact,0,dim))
        fl = tf.sigmoid(_slice(preact, 1, dim))
        fr = tf.sigmoid(_slice(preact, 2, dim))
        o = tf.sigmoid(_slice(preact, 3, dim))
        u = tf.tanh(_slice(preact, 4, dim))

        c_temp=fl*tf.reduce_sum(left_mask[:,:,None]*c_,axis=1)+fr*tf.reduce_sum(right_mask[:,:,None]*c_,axis=1)+i*u
        h_temp=o*tf.tanh(c_temp)
        h=tf.concat([h_[:,:counter_,:],h_temp[:,None,:],h_[:,counter_+1:,:]],axis=1)

        c = tf.concat([c_[:, :counter_, :], c_temp[:,None,:], c_[:, counter_ + 1:, :]], axis=1)
        c_=m_[:,None,None]*c+(1-m_)[:,None,None]*c_
        h_ = m_[:, None, None] * h + (1 - m_)[:, None, None] * h_

        return [h_temp,c_temp]
    s=tf.shape(state_below)
    state_below = tf.matmul(tf.reshape(state_below,[-1,tf.shape(state_below)[-1]]), params[_p(prefix, 'W')]) + params[_p(prefix, 'b')]
    state_below=tf.reshape(state_below,tf.concat([s[:2],[-1]],0))

    rval=tf.scan(_step,
                 [mask,state_below,left_mask,right_mask,tf.range(0,n_steps)],
                 [init_state,init_memory])
    return rval


# input variable
use_noise=tf.placeholder(tf.bool)
word_x1=tf.placeholder(tf.int32,[None,None])
word_x1_mask=tf.placeholder(tf.float32,[None,None])
word_x1_left_mask=tf.placeholder(tf.float32,[None,None,None])
word_x1_right_mask=tf.placeholder(tf.float32,[None,None,None])
word_x2=tf.placeholder(tf.int32,[None,None])
word_x2_mask = tf.placeholder(tf.float32, [None,None])
word_x2_left_mask=tf.placeholder(tf.float32,[None,None,None])
word_x2_right_mask=tf.placeholder(tf.float32,[None,None,None])
y=tf.placeholder(tf.int32,[None])

def build_model(options,params):
    """
    Builds the entire computational graph used for trainning
    :param model_options:
    :return: opt_ret,cost,f_pred,r_prods
    """
    #embedding layer
    with tf.device("/cpu:0"):
        word_embedding_layer=tf.Variable(params['word_embedding'],name='word_embedding')
        emb1=tf.nn.embedding_lookup(word_embedding_layer,word_x1,name='embedding_word_lookup1')
        emb2=tf.nn.embedding_lookup(word_embedding_layer,word_x2,name='embedding_word_lookup2')
    if options['use_dropout']:
        emb1=tf.cond(use_noise,lambda :tf.nn.dropout(emb1,0.5),lambda :emb1)
        emb2 = tf.cond(use_noise, lambda: tf.nn.dropout(emb2, 0.5), lambda: emb2)
    inputs1=(emb1,word_x1_mask,word_x1_left_mask,word_x1_right_mask)
    inputs2 = (emb2, word_x2_mask, word_x2_left_mask, word_x2_right_mask)
    proj1=tree_lstm_layer(params,inputs1,options,prefix='encoder')
    proj2 = tree_lstm_layer(params, inputs2, options, prefix='encoder')
    ctx1=tf.transpose(proj1[0][:,:,:],[1,0,2])
    ctx2 =tf.transpose(proj2[0][:,:,:],[1,0,2])
    ctx1=ctx1*tf.transpose(word_x1_mask[:,:,None],[1,0,2])
    ctx2 = ctx2 * tf.transpose(word_x2_mask[:,:,None],[1,0,2])

    def _step(h,x):
        return tf.matmul(x[0],x[1])
    temp=tf.zeros((tf.shape(ctx1)[0],tf.shape(ctx2)[0]))
    weight_martrix=tf.scan(_step,[ctx1,tf.transpose(ctx2,[0,2,1])],temp)
    weight_martrix_1=tf.exp(weight_martrix)*tf.transpose(word_x2_mask,[1,0])[:,None,:]
    weight_martrix_2 = tf.transpose(tf.exp(weight_martrix) * tf.transpose(word_x1_mask, [1, 0])[:,:,None],[0,2,1])
    weight_martrix_1=weight_martrix_1/tf.reduce_sum(weight_martrix_1,axis=2)[:,:,None]
    weight_martrix_2 = weight_martrix_2 / tf.reduce_sum(weight_martrix_2, axis=2)[:,:,None]
    ctx1_=tf.reduce_sum(weight_martrix_1[:,:,:,None]*ctx2[:,None,:,:],axis=2)
    ctx2_ = tf.reduce_sum(weight_martrix_2[:, :, :, None] * ctx1[:, None, :, :],axis=2)
    inp1=tf.transpose(tf.concat([ctx1, ctx1_, ctx1*ctx1_, ctx1-ctx1_],axis=2),[1,0,2])
    inp2 = tf.transpose(tf.concat([ctx2, ctx2_, ctx2 * ctx2_, ctx2 - ctx2_], axis=2),[1,0,2])

    s=tf.shape(inp1)
    inp1 = tf.nn.relu(tf.matmul(tf.reshape(inp1,[-1,int(inp1.shape[-1])]), params[_p('projection', 'W')]) + params[_p('projection', 'b')])
    inp1=tf.reshape(inp1,tf.concat([s[:2],[-1]],0))
    s=tf.shape(inp2)
    inp2 = tf.nn.relu(tf.matmul(tf.reshape(inp2,[-1,int(inp2.shape[-1])]), params[_p('projection', 'W')]) + params[_p('projection', 'b')])
    inp2=tf.reshape(inp2,tf.concat([s[:2],[-1]],0))

    inputs3 = (inp1, word_x1_mask, word_x1_left_mask, word_x1_right_mask)
    inputs4 = (inp2, word_x2_mask, word_x2_left_mask, word_x2_right_mask)
    proj3=tree_lstm_layer(params,inputs3,options,prefix='decoder')
    proj4 = tree_lstm_layer(params, inputs4, options, prefix='decoder')
    ctx1=proj3[0][:,:,:]
    ctx2 =proj4[0][:,:,:]
    logit0=tf.concat([proj3[0][-1,:,:],proj4[0][-1,:,:]],axis=1)
    logit1=tf.reduce_sum(ctx1*word_x1_mask[:,:,None],axis=0)/tf.reduce_sum(word_x1_mask,axis=0)[:,None]
    logit2 = tf.reduce_max(ctx1 * word_x1_mask[:, :, None], axis=0)
    logit3=tf.reduce_sum(ctx2*word_x2_mask[:,:,None],axis=0)/tf.reduce_sum(word_x2_mask,axis=0)[:,None]
    logit4 = tf.reduce_max(ctx2 * word_x2_mask[:, :, None], axis=0)
    logit=tf.concat([logit0,logit1,logit2,logit3,logit4],axis=1)
    if options['use_dropout']:
        logit=tf.cond(use_noise,lambda :tf.nn.dropout(logit,0.5),lambda :logit)

    logit = tf.nn.tanh(tf.matmul(logit, params[_p('ff_layer_1', 'W')]) + params[_p('ff_layer_1', 'b')])
    if options['use_dropout']:
        logit=tf.cond(use_noise,lambda :tf.nn.dropout(logit,0.5),lambda :logit)
    logit = tf.matmul(logit, params[_p('ff_layer_output', 'W')]) + params[_p('ff_layer_output', 'b')]
    probs=tf.nn.softmax(logit)
    pred=tf.argmax(probs,1)
    cost=tf.losses.sparse_softmax_cross_entropy(y,logit)
    return cost,pred,probs





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
                         batch_size=batch_size,
                         maxlen=model_options['maxlen'])
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
    params=init_params(model_options,worddicts)
    cost,pred,probs=build_model(model_options,params)
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
            _p,_h,_y=prepare_data(x1, x2, label)
            ud_start = time.time()
            _cost,_pred,_prob,_=sess.run([cost,pred,probs,op],feed_dict={use_noise:True,
                                     word_x1:_p[0],word_x1_mask:_p[1],word_x1_left_mask:_p[2],word_x1_right_mask:_p[3],
                                     word_x2: _h[0], word_x2_mask: _h[1],word_x2_left_mask: _h[2],word_x2_right_mask: _h[3],
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
                        _p,_h,_y=prepare_data(x1, x2, label)
                        _cost, _pred, _prob= sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _p[0],
                                                                                              word_x1_mask: _p[1],
                                                                                              word_x1_left_mask: _p[2],
                                                                                              word_x1_right_mask: _p[3],
                                                                                              word_x2: _h[0],
                                                                                              word_x2_mask: _h[1],
                                                                                              word_x2_left_mask: _h[2],
                                                                                              word_x2_right_mask: _h[3],
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
                        _p,_h,_y=prepare_data(x1, x2, label)
                        _cost, _pred, _prob = sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _p[0],
                                                                                              word_x1_mask: _p[1],
                                                                                              word_x1_left_mask: _p[2],
                                                                                              word_x1_right_mask: _p[3],
                                                                                              word_x2: _h[0],
                                                                                              word_x2_mask: _h[1],
                                                                                              word_x2_left_mask: _h[2],
                                                                                              word_x2_right_mask: _h[3],
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
                        _p,_h,_y=prepare_data(x1, x2, label)
                        _cost, _pred, _prob = sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _p[0],
                                                                                              word_x1_mask: _p[1],
                                                                                              word_x1_left_mask: _p[2],
                                                                                              word_x1_right_mask: _p[3],
                                                                                              word_x2: _h[0],
                                                                                              word_x2_mask: _h[1],
                                                                                              word_x2_left_mask: _h[2],
                                                                                              word_x2_right_mask: _h[3],
                                                                                              y: _y
                                                                                              })
                        mismatched_result.extend(_pred)
                        print(len(mismatched_result))
                    except:
                        break
                while True:
                    try:
                        x1, x2, label = test_matched.next()
                        _p,_h,_y=prepare_data(x1, x2, label)
                        _cost, _pred, _prob = sess.run([cost, pred, probs], feed_dict={use_noise: False,
                                                                                              word_x1: _p[0],
                                                                                              word_x1_mask: _p[1],
                                                                                              word_x1_left_mask: _p[2],
                                                                                              word_x1_right_mask: _p[3],
                                                                                              word_x2: _h[0],
                                                                                              word_x2_mask: _h[1],
                                                                                              word_x2_left_mask: _h[2],
                                                                                              word_x2_right_mask: _h[3],
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
