import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import DGMnets
import hessian



# Set random seeds
np.random.seed(42)
tf.set_random_seed(42)


# PDE parameters
r = 0.05            # Interest rate
sigma = 0.25        # Volatility
mu = 0.2            # Mean
lambd = (mu-r)/sigma
gamma = 1           # Utility decay

# TODO: Rename to T1, T2
T0 = 0 + 1e-10      # Initial time
T = 1               # Terminal time

# TODO: Remove S0
S0 = 0.0            # Low boundary
S1 = 0.0 + 1e-10    # Low boundary
S2 = 1              # High boundary



# Merton's analytical known solution
def analytical_solution(t, x):
    return -np.exp(-x*gamma*np.exp(r*(T-t)) - (T-t)*0.5*lambd**2)

def analytical_dVdx(t,x):
    return gamma*np.exp(-0.5*(T-t)*lambd**2 + r*(T-t))*np.exp(-x*gamma*np.exp(r*(T-t)))

def analytical_dV2dxx(t,x):
    return -gamma**2*np.exp(-0.5*(T-t)*lambd**2 + 2*r*(T-t))*np.exp(-x*gamma*np.exp(r*(T-t)))

# Merton's final utility function
def utility(x):
    return -tf.exp(-gamma*x)



# Loss function
def loss(model, t1, x1, t2, x2, t3, x3):
    # Loss term #1: PDE
    V = model(t1, x1)
    V_t = tf.gradients(V, t1)[0]
    V_x = tf.gradients(V, x1)[0]
    V_xx = tf.gradients(V_x, x1)[0]
    f = -0.5*lambd**2*V_x**2 + (V_t + r*x1*V_x)*V_xx

    L1 = tf.reduce_mean(tf.square(f))

    # Loss term #2: boundary condition
    L2 = 0.0
    
    # Loss term #3: initial/terminal condition
    L3 = tf.reduce_mean(tf.square(model(t3, x3) - utility(x3)))

    return (L1, L2, L3)


def functional(model, t, x):
    V = model(t, x)
    V_t = tf.gradients(V, t)[0]
    V_x = tf.gradients(V, x)[0]
    V_xx = hessian.fin_diff_hessian(model, t, x, 1e-4)[:,:,0]
    f = -0.5*lambd**2*V_x**2 + (V_t + r*x*V_x)*V_xx
    return f



# Sampling
# TODO: Use tf.random.uniform instead of np.random.uniform
def sampler(N1, N2, N3):
    # Sampler #1: PDE domain
    t1 = np.random.uniform(low=T0 - 0.5*(T - T0),
                           high=T,
                           size=[N1,1])
    s1 = np.random.uniform(low=S1 - (S2 - S1)*0.5,
                           high=S2 + (S2 - S1)*0.5,
                           size=[N1,1])

    # Sampler #2: boundary condition
    t2 = np.zeros(shape=(1, 1))
    s2 = np.zeros(shape=(1, 1))
    
    # Sampler #3: initial/terminal condition
    t3 = T * np.ones((N3,1)) #Terminal condition
    s3 = np.random.uniform(low=S1 - (S2 - S1)*0.5,
                           high=S2 + (S2 - S1)*0.5,
                           size=[N3,1])
    
    return (t1, s1, t2, s2, t3, s3)



# Neural Network definition
num_layers = 3
nodes_per_layer = 50
model = DGMnets.DGMNet(num_layers, nodes_per_layer)

t1_t = tf.placeholder(tf.float32, [None,1])
x1_t = tf.placeholder(tf.float32, [None,1])
t2_t = tf.placeholder(tf.float32, [None,1])
x2_t = tf.placeholder(tf.float32, [None,1])
t3_t = tf.placeholder(tf.float32, [None,1])
x3_t = tf.placeholder(tf.float32, [None,1])

L1_t, L2_t, L3_t = loss(model, t1_t, x1_t, t2_t, x2_t, t3_t, x3_t)
loss_t = L1_t + L2_t + L3_t

functional_t = functional(model, t1_t, x1_t)


# Optimizer parameters
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_t)


# Training parameters
steps_per_sample = 10
sampling_stages = 400

# Number of samples
NS_1 = 1000
NS_2 = 0
NS_3 = 100


# Plot tensors
tplot_t = tf.placeholder(tf.float32, [None,1], name="tplot_t") # We name to recover it later
xplot_t = tf.placeholder(tf.float32, [None,1], name="xplot_t")
vplot_t = tf.identity(model(tplot_t, xplot_t), name="vplot_t") # Trick for naming the trained model


# Training data holders
sampling_stages_list = []
elapsed_time_list = []
loss_list = []
L1_list = []
L3_list = []


# Train network!!
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for i in range(sampling_stages):
    t1, x1, t2, x2, t3, x3 = sampler(NS_1, NS_2, NS_3)

    start_time = time.clock()
    for _ in range(steps_per_sample):
        loss, L1, L3, _ = sess.run([loss_t, L1_t, L3_t, optimizer],
                               feed_dict = {t1_t:t1, x1_t:x1, t2_t:t2, x2_t:x2, t3_t:t3, x3_t:x3})
    end_time = time.clock()
    elapsed_time = end_time - start_time

    sampling_stages_list.append(i)
    elapsed_time_list.append(elapsed_time)
    loss_list.append(loss)
    L1_list.append(L1)
    L3_list.append(L3)
    
    text = "Stage: {:04d}, Loss: {:e}, L1: {:e}, L3: {:e}, {:f} seconds".format(i, loss, L1, L3, elapsed_time)
    print(text)



# Plot results
N = 41      # Points on plot grid

times_to_plot = [0*T, 0.33*T, 0.66*T, T]
tplot = np.linspace(T0, T, N)
xplot = np.linspace(S0, S2, N)

plt.figure(figsize=(8,7))
i = 1
for t in times_to_plot:
    solution_plot = analytical_solution(t, xplot)

    tt = t*np.ones_like(xplot.reshape(-1,1))
    nn_plot, = sess.run([vplot_t],
                        feed_dict={tplot_t:tt, xplot_t:xplot.reshape(-1,1)})

    plt.subplot(2,2,i)
    plt.plot(xplot, solution_plot, 'b')
    plt.plot(xplot, nn_plot, 'r')

    plt.ylim(-1.1, -0.2)
    plt.xlabel("S")
    plt.ylabel("V")
    plt.title("t = %.2f"%t, loc="left")
    i = i+1

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()


# Save the trained model
saver = tf.train.Saver()
saver.save(sess, './SavedNets/merton_1d_hessian')


# Save the time tracking
np.save('./TimeTracks/merton_1d_hessians',
        (sampling_stages_list, elapsed_time_list, loss_list, L1_list, L3_list))



# Plot losses X stages
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.semilogy(sampling_stages_list,loss_list)
plt.title("Loss", loc="left")
plt.xlabel("Stage")
plt.ylim(1e-8, 1)

plt.subplot(1,3,2)
plt.semilogy(sampling_stages_list,L1_list)
plt.title("L1 loss", loc="left")
plt.xlabel("Stage")
plt.ylim(1e-8, 1)

plt.subplot(1,3,3)
plt.semilogy(sampling_stages_list,L3_list)
plt.title("L3 loss", loc="left")
plt.xlabel("Stage")
plt.ylim(1e-8, 1)

plt.show()
