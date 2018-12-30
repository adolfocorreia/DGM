import tensorflow as tf


## Neural Nets

# Layers used in the architecture
class LSTMLikeLayer(tf.keras.layers.Layer): # Entry layer
    def __init__(self, num_outputs,dimshape=2,
                 trans1 = "tanh",trans2 = "tanh"): # shape[1] of input tensor in dimshape
        super(LSTMLikeLayer,self).__init__()

        self.num_outputs = num_outputs
        self.dimshape = dimshape

        self.Uz = self.add_variable("Uz",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ug = self.add_variable("Ug",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ur = self.add_variable("Ur",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Uh = self.add_variable("Uh",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wz = self.add_variable("Wz",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wg = self.add_variable("Wg",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wr = self.add_variable("Wr",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wh = self.add_variable("Wh",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.bz = self.add_variable("bz",shape=[1,self.num_outputs])
        self.bg = self.add_variable("bg",shape=[1,self.num_outputs])
        self.br = self.add_variable("br",shape=[1,self.num_outputs])
        self.bh = self.add_variable("bh",shape=[1,self.num_outputs])

        if trans1 == "tanh":
            self.trans1 = tf.nn.tanh
        elif trans1 == "relu":
            self.trans1 = tf.nn.relu
        elif trans1 == "sigmoid":
            self.trans1 = tf.nn.sigmoid
        if trans2 == "tanh":
            self.trans2 = tf.nn.tanh
        elif trans2 == "relu":
            self.trans2 = tf.nn.relu
        elif trans2 == "sigmoid":
            self.trans2 = tf.nn.relu

    def build(self,input_shape):
        return
    
    def call(self,input):
        return
    
    def call2(self,S,X): # This is the function to be called
        Z = self.trans1(tf.add(tf.add(tf.matmul(X,self.Uz), tf.matmul(S,self.Wz)), self.bz))
        G = self.trans1(tf.add(tf.add(tf.matmul(X,self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.trans1(tf.add(tf.add(tf.matmul(X,self.Ur), tf.matmul(S, self.Wr)), self.br))
        H = self.trans2(tf.add(tf.add(tf.matmul(X,self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))
        Snew = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z,S))
        return Snew


class LSTMLikeLayerT(tf.keras.layers.Layer): # Entry layer
    def __init__(self, num_outputs,dimshape): # shape[1] of input tensor in dimshape
        super(LSTMLikeLayerT,self).__init__()

        self.num_outputs = num_outputs
        self.dimshape = dimshape

        self.Uz = self.add_variable("Uz",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ug = self.add_variable("Ug",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ur = self.add_variable("Ur",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Uh = self.add_variable("Uh",shape=[self.dimshape,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wz = self.add_variable("Wz",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wg = self.add_variable("Wg",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wr = self.add_variable("Wr",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wh = self.add_variable("Wh",shape=[self.num_outputs,self.num_outputs],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.bz = self.add_variable("bz",shape=[1,self.num_outputs])
        self.bg = self.add_variable("bg",shape=[1,self.num_outputs])
        self.br = self.add_variable("br",shape=[1,self.num_outputs])
        self.bh = self.add_variable("bh",shape=[1,self.num_outputs])
    
    def build(self,input_shape):
        return
    
    def call(self,input):
        return
    
    def call2(self,S,X,S1): # This is the function to be called
        Z = tf.tanh(tf.add(tf.add(tf.matmul(X,self.Uz), tf.matmul(S,self.Wz)), self.bz))
        G = tf.tanh(tf.add(tf.add(tf.matmul(X,self.Ug), tf.matmul(S1,self.Wg)), self.bg))
        R = tf.tanh(tf.add(tf.add(tf.matmul(X,self.Ur), tf.matmul(S,self.Wr)), self.br))
        H = tf.tanh(tf.add(tf.add(tf.matmul(X,self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))
        Snew = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z,S))
        return Snew


class DenseLayer(tf.keras.layers.Layer): # Middle layer
    def __init__(self, num_outputs,dimshape,transformation=None): # shape[1] of input tensor in dimshape
        super(DenseLayer,self).__init__()

        self.num_outputs = num_outputs
        self.dimshape = dimshape
        self.W = self.add_variable("W",shape=[self.dimshape,self.num_outputs],
                                   initializer = tf.contrib.layers.xavier_initializer())
        self.b = self.add_variable("b",shape=[1,self.num_outputs])

        if transformation:
            if transformation == "tanh":
                self.transformation = tf.tanh
            elif transformation == "relu":
                self.transformation = tf.nn.relu
        else:
            self.transformation = transformation
    
    def build(self,input_shape):
        return
    
    def call(self,input):
        return
    
    def call2(self,X): # This is the function to be called
        S = tf.add(tf.matmul(X, self.W), self.b)
        if self.transformation:
            S = self.transformation(S)
        return S


class DenseBatchNorm(tf.keras.layers.Layer):
    def __init__(self, num_outputs,activation="relu"):
        super(DenseBatchNorm, self).__init__()

        self.num_outputs = num_outputs

        if activation == "relu":
            self.activation = tf.nn.relu
        elif activation == "sigmoid":
            self.activation = tf.nn.sigmoid
        elif activation == "tanh":
            self.activation = tf.nn.tanh
        elif activation == "None":
            self.activation = None

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", 
                                        shape=[input_shape[-1].value,self.num_outputs],
                                        initializer=tf.contrib.layers.xavier_initializer())
        self.bias = self.add_variable("bias",shape = [1,self.num_outputs])

    def call(self, input, training = False):
        z = tf.matmul(input, self.kernel) + self.bias
        z = tf.layers.batch_normalization(z, training=training)
        if self.activation:
            return self.activation(z)
        else:
            return z


#Model
class DGMNet(tf.keras.Model):
    def __init__(self,n_h,n_layers,dim,final_trans=None):
        super(DGMNet,self).__init__()
        self.initial_layer = DenseLayer(n_h,dim + 1,transformation = "tanh")
        self.lstmlikelist = []
        self.n_layers = n_layers
        for _ in range(self.n_layers):
            self.lstmlikelist.append(LSTMLikeLayer(n_h,dim + 1))
        self.final_layer = DenseLayer(1,n_h,transformation = final_trans)
        
    def call(self,t,x):
        """Run the model"""
        X = tf.concat([t,x],1)
        S = self.initial_layer.call2(X)
        for i in range(self.n_layers):
            S = self.lstmlikelist[i].call2(S,X)
        result = self.final_layer.call2(S)
        return result
    
    def predict(self,t,x):
        return self.call(t,x)


class DGMNetVectorValued(tf.keras.Model):
    def __init__(self,n_h,n_layers,dim,nout,final_trans=None):
        super(DGMNetVectorValued,self).__init__()
        self.initial_layer = DenseLayer(n_h,dim + 1,transformation = "tanh")
        self.lstmlikelist = []
        self.n_layers = n_layers
        for _ in range(self.n_layers):
            self.lstmlikelist.append(LSTMLikeLayer(n_h,dim + 1))
        self.final_layer = DenseLayer(nout,n_h,transformation = final_trans)
        
    def call(self,t,x):
        """Run the model"""
        X = tf.concat([t,x],1)
        S = self.initial_layer.call2(X)
        for i in range(self.n_layers):
            S = self.lstmlikelist[i].call2(S,X)
        result = self.final_layer.call2(S)
        return result
    
    def predict(self,t,x):
        return self.call(t,x)


class DGMNetT(tf.keras.Model):
    def __init__(self,n_h,n_layers):
        super(DGMNetT,self).__init__()
        self.initial_layer = DenseLayer(n_h,2,transformation = "tanh")
        self.lstmlikelist = []
        self.n_layers = n_layers
        for _ in range(self.n_layers):
            self.lstmlikelist.append(LSTMLikeLayerT(n_h,2))
        self.final_layer = DenseLayer(1,n_h,transformation = None)
        
    def call(self,t,x):
        """Run the model"""
        X = tf.concat([t,x],1)
        S = self.initial_layer.call2(X)
        S1 = tf.identity(S)
        for i in range(self.n_layers):
            S = self.lstmlikelist[i].call2(S,X,S1)
        result = self.final_layer.call2(S)
        return result
    
    def predict(self,t,x):
        return self.call(t,x)

# Classification models
class DGMNetMLP(tf.keras.Model):
    def __init__(self,n_layers,n_h):
        super(DGMNetMLP,self).__init__()
        self.denses_hidden = []
        self.n_layers = n_layers
        self.n_h = n_h
        for _ in range(n_layers):
            self.denses_hidden.append(DenseBatchNorm(n_h,activation="relu"))
        self.dense_out = DenseBatchNorm(1,activation="None")
        
    def call(self,t,x):
        """Run the model"""
        input = tf.concat([t,x],1)
        result = self.denses_hidden[0](input,training = True)
        for i in range(1,self.n_layers):
            result = self.denses_hidden[i](result,training = True)
        result = self.dense_out(result,training = True)
        return result
    
    def predict(self,t,x):
        input = tf.concat([t,x],1)
        result = self.denses_hidden[0](input,training = False)
        for i in range(1,self.n_layers):
            result = self.denses_hidden[i](result,training = False)
        result = self.dense_out(result,training = False)
        return result

