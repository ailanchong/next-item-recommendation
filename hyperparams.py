class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/train.tags.de-en.de'
    target_train = 'corpora/train.tags.de-en.en'
    source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 10 
    num_sampled = 20
    num_blocks = 6 
    num_epochs = 3
    num_heads = 4
    embed_size = 128
    item_size = 10000
    dropout_rate = 0.0
    is_training = True