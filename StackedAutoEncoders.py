# -*- coding: utf-8 -*-

import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from MLP import HiddenLayer
from AutoEncoder import AutoEncoder
from LogisticRegression import LogisticRegression


class StackedAutoEncoders(object):
    def __init__(self, np_rng, theano_rng=None, n_ins=784, hidden_layer_sizes=None, n_outs=10):
        """This function initializes a deep autoencoder neural network with the given parameters. It creates multiple hidden layers with the specified sizes and connects them to form the autoencoder. The function also sets up a logistic regression layer at the end for fine-tuning.
        Parameters:
            - np_rng (numpy.random.RandomState): A random number generator for initializing the weights of the network.
            - theano_rng (theano.tensor.shared_randomstreams.RandomStreams): A random number generator for the autoencoder layers.
            - n_ins (int): Number of input units.
            - hidden_layer_sizes (list of ints): Sizes of the hidden layers. If not specified, default values of [500, 500] are used.
            - n_outs (int): Number of output units.
        Returns:
            - None
        Processing Logic:
            - If hidden_layer_sizes is not specified, default values are used.
            - The sigmoid_layers, dA_layers, and params lists are initialized.
            - The number of layers is calculated and asserted to be greater than 0.
            - If theano_rng is not specified, a random number generator is created.
            - The input and output variables are created.
            - For each layer, the input and output sizes are determined and a HiddenLayer and AutoEncoder object are created.
            - The parameters of each layer are added to the params list.
            - A logistic regression layer is created and its parameters are added to the params list.
            - The cost and error functions are calculated for the logistic regression layer."""
        
        hidden_layer_sizes = [500, 500] if hidden_layer_sizes is None else hidden_layer_sizes
        
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)
        
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
     
        self.x = T.matrix('x') 
        self.y = T.ivector('y') 
        
        for i in xrange(self.n_layers):
            if i == 0:
                n_in = n_ins
                layer_input = self.x
            else:
                n_in = hidden_layer_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output

            n_out = hidden_layer_sizes[i]            
            
            sigmoid_layer = HiddenLayer(np_rng, layer_input, n_in, n_out, activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            
            self.params.extend(sigmoid_layer.params)
            
            dA_layer = AutoEncoder(np_rng, n_in, n_out, theano_rng=theano_rng, input=layer_input, 
                                   W=sigmoid_layer.W, b_hid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
            
        self.log_layer = LogisticRegression(self.sigmoid_layers[-1].output, self.y, hidden_layer_sizes[-1], n_outs)
        self.params.extend(self.log_layer.params)

        self.finetune_cost = self.log_layer.negative_log_likelihood()
        self.errors = self.log_layer.errors()        
        
        
    def pretraining_functions(self, train_set_x, batch_size):
        """This function creates pretraining functions for a deep autoencoder model.
        Parameters:
            - self (object): The deep autoencoder model.
            - train_set_x (numpy array): The training dataset.
            - batch_size (int): The batch size for training.
        Returns:
            - pretrain_fns (list): A list of pretraining functions for each layer of the deep autoencoder model.
        Processing Logic:
            - Creates a Theano scalar variable for the batch index.
            - Creates Theano scalar variables for the corruption level and learning rate.
            - Defines the beginning and end of each batch.
            - Initializes an empty list for the pretraining functions.
            - Loops through each layer of the deep autoencoder model.
            - Calculates the cost and updates for each layer.
            - Creates a Theano function for each layer with inputs, outputs, and updates.
            - Appends the function to the pretrain_fns list.
            - Returns the list of pretraining functions."""
        
        
        index = T.lscalar(name='index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(learning_rate, corruption_level)
            fn =  theano.function(inputs=[index, theano.Param(corruption_level, default=0.2), theano.Param(learning_rate, default=0.1)],
                                          outputs=[cost],
                                          updates = updates,
                                          givens={self.x:train_set_x[batch_begin:batch_end]})
                
            pretrain_fns.append(fn)
            
        return pretrain_fns
        
        
    def finetuning_functions(self, datasets, batch_size, learning_rate):
        """Finetunes the parameters of a neural network using a given dataset, batch size, and learning rate.
        Parameters:
            - datasets (tuple): A tuple containing the training, validation, and test sets.
            - batch_size (int): The size of each batch used for training.
            - learning_rate (float): The learning rate used for updating the parameters.
        Returns:
            - train_fn (theano.function): A function that performs one iteration of training on the given dataset.
            - valid_score (list): A list of validation errors for each batch.
            - test_score (list): A list of test errors for each batch.
        Processing Logic:
            - Retrieves the training, validation, and test sets from the given dataset.
            - Calculates the number of batches for the validation and test sets.
            - Defines a function for updating the parameters using the given learning rate.
            - Defines functions for calculating the errors on the validation and test sets.
            - Defines functions for calculating the overall validation and test scores.
            - Returns the training function, validation scores, and test scores."""
        
        
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        
        index = T.lscalar('index')
        
        gparams = T.grad(self.finetune_cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))
            
        train_fn = theano.function(inputs=[index],
                                   outputs=self.finetune_cost,
                                   updates=updates,
                                   givens={self.x: train_set_x[index * batch_size: (index+1) * batch_size],
                                           self.y: train_set_y[index * batch_size: (index+1) * batch_size]})

        test_score_i = theano.function(inputs=[index], 
                                       outputs=self.errors,
                                       givens={self.x: test_set_x[index * batch_size: (index+1) * batch_size],
                                               self.y: test_set_y[index * batch_size: (index+1) * batch_size]})

        valid_score_i = theano.function(inputs=[index], 
                                        outputs=self.errors,
                                        givens={self.x: valid_set_x[index * batch_size: (index+1) * batch_size],
                                                self.y: valid_set_y[index * batch_size: (index+1) * batch_size]})

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score   
            
            
            
            
            
            
