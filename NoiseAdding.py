import numpy as np
from sklearn.utils import shuffle
from random import choice
from numpy.testing import assert_array_almost_equal




def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print( np.max(y), P.shape[0])
    assert( P.shape[0] == P.shape[1])
    assert( np.max(y) < P.shape[0])

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
#    assert ((P >= 0.0).all())

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert( actual_noise > 0.0)
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print( P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print( P)

    return y_train, actual_noise

def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate




def corruptingDataXX(x, y, percent , noise_type):
    
        edge = (len(y)/100)*percent
        edge = int(edge)

        for i in range(0,edge):

            n = y[i,0]
            # prepare a sequence
            n = int(n)
            r = list(range(0,n)) + list(range(n+1, 10))
            selectedLabel = choice(r)
            y[i,0] = selectedLabel
            y[i,1] = 0

        #make shuffle after corrupting some instances
        x, y = shuffle(x, y, random_state=0)
        print(percent,"% of the data has been corrupted")
        return x, y

def corruptingData(x, y, noise_type, noise_rate):
    c=0
    train_labels = y
    train_labels=np.asarray([[train_labels[i]] for i in range(len(train_labels))])
    train_noisy_labels, actual_noise_rate = noisify(dataset='mnist', train_labels=train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=None, nb_classes=10)
    train_noisy_labels=[i[0] for i in train_noisy_labels]
    _train_labels=[i[0] for i in train_labels]
    noise_or_not = np.transpose(train_noisy_labels)==np.transpose(_train_labels)
    
    yy = train_noisy_labels
    yy = np.asarray(yy)
    onesValues = np.ones((yy.shape[0],1))
    yy = yy.reshape(yy.shape[0],1)
    y_copy = np.append(yy, onesValues, axis=1)
    
    assert(len(y_copy) == len(noise_or_not))
    for i in range(len(y_copy)):
        if noise_or_not[i] == False:
            y_copy[i,1] = 0
            c+=1
            
    print("c :",c)
    print("y_copy :", y_copy.shape[0])
    x, y_copy = shuffle(x, y_copy, random_state=0)
    implemented_noise_rate = (c/y_copy.shape[0]*100)
    print(implemented_noise_rate,"% of the data has been corrupted")
    return x, y_copy
                
