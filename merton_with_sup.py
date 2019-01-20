import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import DGMnets



# Set random seeds
np.random.seed(42)
tf.set_random_seed(42)


# Loss for heat equation
gamma = 1.0
mu = 0.2
sigma = 0.25
r = 0.05

def analytical_solution(t,x):
    lambd = (mu - r)/sigma
    return -np.exp(-x*gamma*np.exp(r*(1.0 - t)) - (1.0 - t)*0.5*lambd**2)

def loss_solver(solver,maximizer,t1,x1,t3,x3):
    # Loss term 1: PDE
    u1 = solver(t1,x1)
    u1_t = tf.gradients(u1,t1)[0]
    u1_x = tf.gradients(u1,x1)[0]
    u1_xx = tf.gradients(u1_x,x1)[0]
    pi = maximizer(t1,x1)
    f = u1_t + pi*(mu-r)*u1_x + r*x1*u1_x + 0.5*sigma**2*pi**2*u1_xx
    L1 = tf.reduce_mean(tf.square(f))
    # Loss term 2: Initial condition
    u3 = solver(t3,x3)
    terminal_condition = -tf.exp(-gamma*x3)
    L3 = tf.reduce_mean(tf.square(u3 - terminal_condition))
    return L1,L3

def loss_maximizer(maximizer,solver,t1,x1):
    u1 = solver(t1,x1)
    u1_x = tf.gradients(u1,x1)[0]
    u1_xx = tf.gradients(u1_x,x1)[0]
    pi = maximizer(t1,x1)
    f = pi*(mu-r)*u1_x + r*x1*u1_x + 0.5*sigma**2*pi**2*u1_xx
    L = tf.reduce_mean(f)
    return L

def sampler(N1,N3,tfinal,xlow,xhigh):
    # N2 has to be pair
    # Sampler #1: PDE domain
    t1 = np.random.uniform(low=0.0,high=tfinal,size=[N1,1])
    x1 = np.random.uniform(low=xlow,high=xhigh,size=[N1,1])
    
    # Sampler #3: initial/terminal condition
    t3 = tfinal*np.ones([N3,1])
    x3 = np.random.uniform(low=xlow,high=xhigh,size=[N3,1])
    
    return t1,x1,t3,x3

with tf.variable_scope("solver",reuse=True):
    solver = DGMnets.DGMNet(3,50)
with tf.variable_scope("maximizer",reuse=True):
    maximizer = DGMnets.DGMNet(3,50)

print("Net created")
# Loss tensors
t1_t = tf.placeholder(tf.float32,[None,1])
x1_t = tf.placeholder(tf.float32,[None,1])
t3_t = tf.placeholder(tf.float32,[None,1])
x3_t = tf.placeholder(tf.float32,[None,1])
L1_t,L3_t = loss_solver(solver,maximizer,t1_t,x1_t,t3_t,x3_t)
loss_solver_t = L1_t + L3_t
t4_t = tf.placeholder(tf.float32,[None,1])
x4_t = tf.placeholder(tf.float32,[None,1])
loss_maximizer_t = loss_maximizer(maximizer,solver,t4_t,x4_t)

# Optimizer
solver_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="solver")
solver_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).\
                        minimize(loss_solver_t,var_list=solver_vars)
maximizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="maximizer")
maximizer_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).\
                        minimize(loss_maximizer_t,var_list=maximizer_vars)

print("Optimizer set")
# Training parameters
steps_per_sample = 10
sampling_stages = 500

samples_1 = 100
samples_3 = 100


# Train network!!
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
print("Session done")
for i in range(sampling_stages):
    t1,x1,t3,x3 = sampler(samples_1,samples_3,1.0,0.0,1.0)
    for _ in range(steps_per_sample):
        loss_solver,_ = sess.run([loss_solver_t,solver_optimizer],
                                feed_dict = {t1_t:t1,x1_t:x1,t3_t:t3,x3_t:x3})
    for _ in range(steps_per_sample):
        loss_maximizer,_ = sess.run([loss_maximizer_t,maximizer_optimizer],
                                        feed_dict = {t4_t:t1,x4_t:x1})
    print("Solver: %f, Maximizer: %f, step %i"%(loss_solver,loss_maximizer,i))

#Checking results
plot_t = solver(t1_t,x1_t)

nplot = 41
plot_times = [0.25,0.5,0.75,1.0]
xplot = np.linspace(0.0,1.0,nplot).reshape(-1,1)

plt.figure(figsize=(8,7))
i = 1
for t in plot_times:
    tt = t*np.ones_like(xplot)
    nn_plot, = sess.run([plot_t],feed_dict={t1_t:tt,x1_t:xplot})

    plt.subplot(2,2,i)
    plt.plot(xplot,nn_plot,'b--')
    plt.plot(xplot,analytical_solution(t,xplot),'r-')

    i = i + 1

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()

print("End")
