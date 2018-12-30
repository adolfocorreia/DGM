import tensorflow as tf


# Finite Difference Hessian functions

def canonical_vector(n, j):
    # Return e_j in Rn
    return tf.reshape(tf.one_hot(j, depth=n), (1,-1))

def fin_diff(f, t, x, delta, j):
    # Get grad(f(x + delta*ej)) - grad(f(x - delta*ej)) / 2*delta
    ej = canonical_vector(x.get_shape().as_list()[1], j)
    xej_plus = x + delta*ej
    xej_minus = x - delta*ej
    f_plus = f(t, xej_plus)
    f_minus = f(t, xej_minus)
    df_plus = tf.gradients(f_plus, xej_plus)[0]
    df_minus = tf.gradients(f_minus, xej_minus)[0]
    return (df_plus - df_minus) / (2*delta)

def fin_diff_hessian(f, t, x, delta):
    # Get the approximation of the Hessian as finite differences. Shape = [?,ndims,ndims]
    n = tf.shape(x)[1]
    array = tf.TensorArray(x.dtype,size=tf.shape(x)[1])
    loop_vars = [tf.constant(0), array] # [counter, result_holder]
    _, hessian = tf.while_loop(lambda j, _: j < n,
                               lambda j, result: (j+1, result.write(j, fin_diff(f, t, x, delta, j))),
                               loop_vars)
    H = hessian.stack()
    H = tf.transpose(H, perm=[1,0,2])
    H = 0.5*(H + tf.transpose(H, perm=[0,2,1]))
    return H
