import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import DGMnets



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


# Loss relative to the value function
def loss_value_function(value_function, optimal_control, t1, x1, t3, x3):
    # Loss term 1: PDE
    u1 = value_function(t1,x1)
    u1_t = tf.gradients(u1,t1)[0]
    u1_x = tf.gradients(u1,x1)[0]
    u1_xx = tf.gradients(u1_x,x1)[0]
    pi = optimal_control(t1,x1)
    f = u1_t + pi*(mu-r)*u1_x + r*x1*u1_x + 0.5*sigma**2*pi**2*u1_xx
    L1 = tf.reduce_mean(tf.square(f))

    # Loss term 2: Initial condition
    u3 = value_function(t3,x3)
    terminal_condition = -tf.exp(-gamma*x3)
    L3 = tf.reduce_mean(tf.square(u3 - terminal_condition))

    return L1,L3


# Loss relative to the optimal control function
def loss_optimal_control(value_function, optimal_control, t1, x1):
    u1 = value_function(t1,x1)
    u1_x = tf.gradients(u1,x1)[0]
    u1_xx = tf.gradients(u1_x,x1)[0]
    pi = optimal_control(t1,x1)
    f = pi*(mu-r)*u1_x + r*x1*u1_x + 0.5*sigma**2*pi**2*u1_xx
    L = tf.reduce_mean(f)
    return L


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


# Neural Networks definition
num_layers = 3
nodes_per_layer = 50

# Value function net
with tf.variable_scope("value_function", reuse=True):
    value_function = DGMnets.DGMNet(num_layers, nodes_per_layer)
# Optimal control function net
with tf.variable_scope("optimal_control", reuse=True):
    optimal_control = DGMnets.DGMNet(num_layers, nodes_per_layer)

# Nets' input placeholders
t1_t = tf.placeholder(tf.float32, [None,1])
x1_t = tf.placeholder(tf.float32, [None,1])
t3_t = tf.placeholder(tf.float32, [None,1])
x3_t = tf.placeholder(tf.float32, [None,1])

t4_t = tf.placeholder(tf.float32, [None,1])
x4_t = tf.placeholder(tf.float32, [None,1])

# Loss terms
L1_t, L3_t = loss_value_function(value_function, optimal_control,
                                 t1_t, x1_t, t3_t, x3_t)
loss_value_function_t = L1_t + L3_t

loss_optimal_control_t = loss_optimal_control(value_function, optimal_control,
                                              t4_t, x4_t)


# Optimizers
value_function_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="value_function")
value_function_optimizer = tf.train.AdamOptimizer(learning_rate=0.001). \
                           minimize(loss_value_function_t, var_list=value_function_vars)

optimal_control_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="optimal_control")
optimal_control_optimizer = tf.train.AdamOptimizer(learning_rate=0.001). \
                            minimize(-loss_optimal_control_t, var_list=optimal_control_vars)


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
vplot_t = tf.identity(value_function(tplot_t, xplot_t), name="vplot_t") # Trick for naming the trained model


# Training data holders
sampling_stages_list = []
elapsed_time_list = []
LVF_list = []
LOC_list = []


# Train network!!
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for i in range(sampling_stages):
    t1, x1, _, _, t3, x3 = sampler(NS_1, NS_2, NS_3)

    start_time = time.clock()
    for _ in range(steps_per_sample):
        loss_value_function, _ = sess.run([loss_value_function_t, value_function_optimizer],
                                          feed_dict = {t1_t:t1, x1_t:x1, t3_t:t3, x3_t:x3})
    for _ in range(steps_per_sample):
        loss_optimal_control, _ = sess.run([loss_optimal_control_t, optimal_control_optimizer],
                                           feed_dict = {t4_t:t1, x4_t:x1})
    end_time = time.clock()
    elapsed_time = end_time - start_time

    sampling_stages_list.append(i)
    elapsed_time_list.append(elapsed_time)
    LVF_list.append(loss_value_function)
    LOC_list.append(loss_optimal_control)
    
    text = "Stage: {:04d}, LVF: {:e}, LOC: {:e}, {:f} seconds".format(i, loss_value_function, loss_optimal_control, elapsed_time)
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
saver.save(sess, './SavedNets/merton_with_sup')


# Save the time tracking
np.save('./TimeTracks/merton_with_sup',
        (sampling_stages_list, elapsed_time_list, LVF_list, LOC_list))



# Plot losses X stages
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.semilogy(sampling_stages_list,LVF_list)
plt.title("Value Function Loss", loc="left")
plt.xlabel("Stage")
plt.ylim(1e-8, 1)

plt.subplot(1,2,2)
plt.semilogy(sampling_stages_list,LOC_list)
plt.title("Optimal Control Loss", loc="left")
plt.xlabel("Stage")
plt.ylim(1e-2, 1)

plt.show()
