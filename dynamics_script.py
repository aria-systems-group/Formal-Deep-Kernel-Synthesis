from import_script import *

# ======================================================================
# 1. Define dynamics function
# ======================================================================


def g_lin1(x):
    A = np.array([[0.9, -0.4], [0.4, 0.5]])
    result = A @ x
    return result


def g_lin2(x):
    A = np.array([[0.8, 0.5], [0, 0.5]])
    result = A @ x
    return result


def g_lin3(x):
    A = np.array([[0.5, 0], [-0.5, 0.8]])
    result = A @ x
    return result


def g_nonlinear1(x):
    return [x[0] - 0.05 * x[1], x[1] + 0.1 * math.sin(x[0])]


def g_3d_rot(x):
    A = np.array([[0.9, -0.4, 0.0], [0.4, 0.5, 0.0], [0.0, 0.0, -0.05]])
    result = A @ x
    return result


def g_5d_mode0(xs):

    result = [xs[0] + 0.35 + (0.1 * np.sin(xs[1])),
              xs[1] + (0.15 * np.cos(xs[0])) + (0.05 * xs[2]),
              (0.3 * xs[2]) + (0.4 * xs[3]),
              (0.4 * xs[3]) + (0.05 * xs[4]),
              (0.5 * xs[4])]

    return result


def g_5d_mode1(xs):

    result = [xs[0] - 0.35 + (0.1 * np.sin(xs[1])),
              xs[1] + (0.15 * np.cos(xs[0])) + (0.05 * xs[2]),
              (0.3 * xs[2]) + (0.4 * xs[3]),
              (0.4 * xs[3]) + (0.05 * xs[4]),
              (0.5 * xs[4])]

    return result


def g_5d_mode2(xs):

    result = [xs[0] + (0.15 * np.cos(xs[1])),
              xs[1] + 0.35 + (0.1 * np.sin(xs[0])) + (0.05 * xs[2]),
              (0.3 * xs[2]) + (0.4 * xs[3]),
              (0.4 * xs[3]) + (0.05 * xs[4]),
              (0.5 * xs[4])]

    return result


def g_5d_mode3(xs):

    result = [xs[0] + (0.15 * np.cos(xs[1])),
              xs[1] - 0.35 + (0.1 * np.sin(xs[0])) + (0.05 * xs[2]),
              (0.3 * xs[2]) + (0.4 * xs[3]),
              (0.4 * xs[3]) + (0.05 * xs[4]),
              (0.5 * xs[4])]

    return result


def g_sin(x):
    return np.sin(x)


def crazy_1d(x):
    x_scale = [x_/5. for x_ in x]
    part_1 = x * np.sin(x_scale)
    x_scale = [x_/15. for x_ in x]
    part_2 = 3.* np.cos(x_scale)
    x_scale = [-x_ for x_ in x]
    part_3 = 2. * np.exp(x_scale)
    return part_1 + part_2 + part_3


def g_x_sin(x):
    return x*np.sin(x)


def g_2d_mode0(x):
    result = [x[0] + 0.25 + 0.05*np.sin(x[1]),
              x[1] + 0.1*np.cos(x[0])]
    return result


def g_2d_mode1(x):
    result = [x[0] + -0.25 + 0.05*np.sin(x[1]),
              x[1] + 0.1*np.cos(x[0])]
    return result


def g_2d_mode2(x):
    result = [x[0] + 0.1*np.cos(x[1]),
              x[1] + 0.25 + 0.05*np.sin(x[0])]
    return result


def g_2d_mode3(x):
    result = [x[0] + 0.1*np.cos(x[1]),
              x[1] + -0.25 + 0.05*np.sin(x[0])]
    return result


dubins_ts = 0.1
print(f'Dubins Ts = {dubins_ts}')
def g_3d_mode1(xs, Ts=dubins_ts):
    u = 10
    omega = 5
    phi = -0.3

    result = [xs[0] + Ts * u * np.cos(xs[2]),
              xs[1] + Ts * u * np.sin(xs[2]),
              xs[2] + (phi - xs[2]) * Ts * omega]
    return result


def g_3d_mode2(xs, Ts=dubins_ts):
    u = 10
    omega = 5
    phi = -0.15

    result = [xs[0] + Ts * u * np.cos(xs[2]),
              xs[1] + Ts * u * np.sin(xs[2]),
              xs[2] + (phi - xs[2]) * Ts * omega]
    return result


def g_3d_mode3(xs, Ts=dubins_ts):
    u = 10
    omega = 5
    phi = 0.

    result = [xs[0] + Ts * u * np.cos(xs[2]),
              xs[1] + Ts * u * np.sin(xs[2]),
              xs[2] + (phi - xs[2]) * Ts * omega]
    return result


def g_3d_mode4(xs, Ts=dubins_ts):
    u = 10
    omega = 5
    phi = 0.3

    result = [xs[0] + Ts * u * np.cos(xs[2]),
              xs[1] + Ts * u * np.sin(xs[2]),
              xs[2] + (phi - xs[2]) * Ts * omega]
    return result


def g_3d_mode5(xs, Ts=dubins_ts):
    u = 10
    omega = 5
    phi = 0.15

    result = [xs[0] + Ts * u * np.cos(xs[2]),
              xs[1] + Ts * u * np.sin(xs[2]),
              xs[2] + (phi - xs[2]) * Ts * omega]
    return result


def g_3d_mode6(xs, Ts=dubins_ts):
    u = 10
    omega = 5
    phi = 0.45

    result = [xs[0] + Ts * u * np.cos(xs[2]),
              xs[1] + Ts * u * np.sin(xs[2]),
              xs[2] + (phi - xs[2]) * Ts * omega]
    return result


def g_3d_mode7(xs, Ts=dubins_ts):
    u = 10
    omega = 5
    phi = -0.45

    result = [xs[0] + Ts * u * np.cos(xs[2]),
              xs[1] + Ts * u * np.sin(xs[2]),
              xs[2] + (phi - xs[2]) * Ts * omega]
    return result


def known_part_3d(x):
    return x


def g_unicycle_mode0(x):
    v = 0.5
    dt = 0.5
    theta_dot = 0.0

    k1 = [x[2],
          x[3],
          -v * np.sin(x[4]) * theta_dot,
          v * np.cos(x[4]) * theta_dot,
          theta_dot]

    k2 = [x[2] + dt*k1[2]/2.0,
          x[3] + dt*k1[3]/2.0,
          -v * np.sin(x[4] + dt*k1[4]/2.0) * theta_dot,
          v * np.cos(x[4] + dt*k1[4]/2.0) * theta_dot,
          theta_dot]

    k3 = [x[2] + dt*k2[2]/2.0,
          x[3] + dt*k2[3]/2.0,
          -v * np.sin(x[4] + dt*k2[4]/2.0) * theta_dot,
          v * np.cos(x[4] + dt*k2[4]/2.0) * theta_dot,
          theta_dot]

    k4 = [x[2] + dt*k3[2],
          x[3] + dt*k3[3],
          -v * np.sin(x[4] + dt*k3[4]) * theta_dot,
          v * np.cos(x[4] + dt*k3[4]) * theta_dot,
          theta_dot]

    result = [x[0] + 1.0/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt,
              x[1] + 1.0/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt,
              x[2] + 1.0/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt,
              x[3] + 1.0/6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) * dt,
              x[4] + 1.0/6.0 * (k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4]) * dt]

    return result


def g_unicycle_mode1(x):
    v = 0.5
    dt = 0.5
    theta_dot = -0.25

    k1 = [x[2],
          x[3],
          -v * np.sin(x[4]) * theta_dot,
          v * np.cos(x[4]) * theta_dot,
          theta_dot]

    k2 = [x[2] + dt*k1[2]/2.0,
          x[3] + dt*k1[3]/2.0,
          -v * np.sin(x[4] + dt*k1[4]/2.0) * theta_dot,
          v * np.cos(x[4] + dt*k1[4]/2.0) * theta_dot,
          theta_dot]

    k3 = [x[2] + dt*k2[2]/2.0,
          x[3] + dt*k2[3]/2.0,
          -v * np.sin(x[4] + dt*k2[4]/2.0) * theta_dot,
          v * np.cos(x[4] + dt*k2[4]/2.0) * theta_dot,
          theta_dot]

    k4 = [x[2] + dt*k3[2],
          x[3] + dt*k3[3],
          -v * np.sin(x[4] + dt*k3[4]) * theta_dot,
          v * np.cos(x[4] + dt*k3[4]) * theta_dot,
          theta_dot]

    result = [x[0] + 1.0/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt,
              x[1] + 1.0/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt,
              x[2] + 1.0/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt,
              x[3] + 1.0/6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) * dt,
              x[4] + 1.0/6.0 * (k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4]) * dt]

    return result


def g_unicycle_mode2(x):
    v = 0.5
    dt = 0.5
    theta_dot = 0.25

    k1 = [x[2],
          x[3],
          -v * np.sin(x[4]) * theta_dot,
          v * np.cos(x[4]) * theta_dot,
          theta_dot]

    k2 = [x[2] + dt*k1[2]/2.0,
          x[3] + dt*k1[3]/2.0,
          -v * np.sin(x[4] + dt*k1[4]/2.0) * theta_dot,
          v * np.cos(x[4] + dt*k1[4]/2.0) * theta_dot,
          theta_dot]

    k3 = [x[2] + dt*k2[2]/2.0,
          x[3] + dt*k2[3]/2.0,
          -v * np.sin(x[4] + dt*k2[4]/2.0) * theta_dot,
          v * np.cos(x[4] + dt*k2[4]/2.0) * theta_dot,
          theta_dot]

    k4 = [x[2] + dt*k3[2],
          x[3] + dt*k3[3],
          -v * np.sin(x[4] + dt*k3[4]) * theta_dot,
          v * np.cos(x[4] + dt*k3[4]) * theta_dot,
          theta_dot]

    result = [x[0] + 1.0/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt,
              x[1] + 1.0/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt,
              x[2] + 1.0/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt,
              x[3] + 1.0/6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) * dt,
              x[4] + 1.0/6.0 * (k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4]) * dt]

    return result


def g_unicycle_4d_mode0(x):
    # states are x, y, theta, phi, phi being a steering angle and phi_dot being the rate
    v = 1.2
    L = 0.5
    dt = 0.5
    phi_dot = 0.0

    k1 = [v * np.cos(x[2]),
          v * np.sin(x[2]),
          v/L * np.tan(x[3]),
          phi_dot]

    k2 = [v * np.cos(x[2] + dt*k1[2]/2.0),
          v * np.sin(x[2] + dt*k1[2]/2.0),
          v/L * np.tan(x[3] + dt*k1[3]/2.0),
          phi_dot]

    k3 = [v * np.cos(x[2] + dt * k2[2] / 2.0),
          v * np.sin(x[2] + dt * k2[2] / 2.0),
          v/L * np.tan(x[3] + dt * k2[3] / 2.0),
          phi_dot]

    k4 = [v * np.cos(x[2] + dt * k3[2]),
          v * np.sin(x[2] + dt * k3[2]),
          v/L * np.tan(x[3] + dt * k3[3]),
          phi_dot]

    result = [x[0] + 1.0/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt,
              x[1] + 1.0/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt,
              x[2] + 1.0/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt,
              x[3] + 1.0/6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) * dt]

    return result


def g_unicycle_4d_mode1(x):
    v = 1.2
    L = 0.5
    dt = 0.5
    phi_dot = -0.5

    k1 = [v * np.cos(x[2]),
          v * np.sin(x[2]),
          v/L * np.tan(x[3]),
          phi_dot]

    k2 = [v * np.cos(x[2] + dt*k1[2]/2.0),
          v * np.sin(x[2] + dt*k1[2]/2.0),
          v/L * np.tan(x[3] + dt*k1[3]/2.0),
          phi_dot]

    k3 = [v * np.cos(x[2] + dt * k2[2] / 2.0),
          v * np.sin(x[2] + dt * k2[2] / 2.0),
          v/L * np.tan(x[3] + dt * k2[3] / 2.0),
          phi_dot]

    k4 = [v * np.cos(x[2] + dt * k3[2]),
          v * np.sin(x[2] + dt * k3[2]),
          v/L * np.tan(x[3] + dt * k3[3]),
          phi_dot]

    result = [x[0] + 1.0/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt,
              x[1] + 1.0/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt,
              x[2] + 1.0/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt,
              x[3] + 1.0/6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) * dt]

    return result


def g_unicycle_4d_mode2(x):
    v = 1.2
    L = 0.5
    dt = 0.5
    phi_dot = .5

    k1 = [v * np.cos(x[2]),
          v * np.sin(x[2]),
          v/L * np.tan(x[3]),
          phi_dot]

    k2 = [v * np.cos(x[2] + dt*k1[2]/2.0),
          v * np.sin(x[2] + dt*k1[2]/2.0),
          v/L * np.tan(x[3] + dt*k1[3]/2.0),
          phi_dot]

    k3 = [v * np.cos(x[2] + dt * k2[2] / 2.0),
          v * np.sin(x[2] + dt * k2[2] / 2.0),
          v/L * np.tan(x[3] + dt * k2[3] / 2.0),
          phi_dot]

    k4 = [v * np.cos(x[2] + dt * k3[2]),
          v * np.sin(x[2] + dt * k3[2]),
          v/L * np.tan(x[3] + dt * k3[3]),
          phi_dot]

    result = [x[0] + 1.0/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt,
              x[1] + 1.0/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt,
              x[2] + 1.0/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt,
              x[3] + 1.0/6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) * dt]

    return result


def unicycle_left(xs):
    x = xs[0]
    y = xs[1]
    x_dot = xs[2]
    y_dot = xs[3]

    dir_ = np.array([x_dot, y_dot])
    v = np.linalg.norm(dir_, ord=2)

    theta = math.atan2(y_dot, x_dot)
    theta_des = math.pi / 4.5
    des_theta = theta + theta_des

    omega = 0.5
    u2 = omega * v * np.cos(theta)
    u1 = -u2 * np.tan(theta)

    state = [x,
             y,
             v,
             theta]

    dt = 2.0

    k1 = [v * np.cos(theta),
          v * np.sin(theta),
          u1 * np.cos(theta) + u2 * np.sin(theta),
          (- np.sin(theta) * u2 + np.cos(theta) * u1) / v]

    new_theta = theta + (dt / 2.) * k1[3]
    u2 = omega * v * np.cos(new_theta)
    u1 = -u2 * np.tan(new_theta)

    k2 = [(v + (dt / 2.) * k1[2]) * np.cos(new_theta),
          (v + (dt / 2.) * k1[2]) * np.sin(new_theta),
          u1 * np.cos(new_theta) + u2 * np.sin(new_theta),
          (- np.sin(new_theta) * u1 + np.cos(new_theta) * u2) / (v + (dt / 2.) * k1[2])]

    new_theta = theta + (dt / 2.) * k2[3]
    u2 = omega * v * np.cos(new_theta)
    u1 = -u2 * np.tan(new_theta)

    k3 = [(v + (dt / 2.) * k2[2]) * np.cos(new_theta),
          (v + (dt / 2.) * k2[2]) * np.sin(new_theta),
          u1 * np.cos(new_theta) + u2 * np.sin(new_theta),
          (- np.sin(new_theta) * u1 + np.cos(new_theta) * u2) / (v + (dt / 2.) * k2[2])]

    new_theta = theta + dt * k3[3]
    u2 = omega * v * np.cos(new_theta)
    u1 = -u2 * np.tan(new_theta)

    k4 = [(v + dt * k3[2]) * np.cos(new_theta),
          (v + dt * k3[2]) * np.sin(new_theta),
          u1 * np.cos(new_theta) + u2 * np.sin(new_theta),
          (- np.sin(new_theta) * u1 + np.cos(new_theta) * u2) / (v + dt * k3[2])]

    new_x = [state[0] + 1.0 / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * dt,
             state[1] + 1.0 / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * dt,
             state[2] + 1.0 / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) * dt,
             state[3] + 1.0 / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) * dt]

    x_new = new_x[0]
    y_new = new_x[1]
    x_dot_new = new_x[2] * np.cos(new_x[3])
    y_dot_new = new_x[2] * np.sin(new_x[3])

    result = [x_new, y_new, x_dot_new, y_dot_new]

    return result


def unicycle_right(xs):
    x = xs[0]
    y = xs[1]
    x_dot = xs[2]
    y_dot = xs[3]

    dir_ = np.array([x_dot, y_dot])
    v = np.linalg.norm(dir_, ord=2)

    theta = math.atan2(y_dot, x_dot)
    theta_des = -math.pi / 4.5
    des_theta = theta + theta_des

    omega = -0.5
    u2 = omega * v * np.cos(theta)
    u1 = -u2 * np.tan(theta)

    state = [x,
             y,
             v,
             theta]

    dt = 2.0

    k1 = [v * np.cos(theta),
          v * np.sin(theta),
          u1 * np.cos(theta) + u2 * np.sin(theta),
          (- np.sin(theta) * u2 + np.cos(theta) * u1) / v]

    new_theta = theta + (dt / 2.) * k1[3]
    u2 = omega * v * np.cos(new_theta)
    u1 = -u2 * np.tan(new_theta)

    k2 = [(v + (dt / 2.) * k1[2]) * np.cos(new_theta),
          (v + (dt / 2.) * k1[2]) * np.sin(new_theta),
          u1 * np.cos(new_theta) + u2 * np.sin(new_theta),
          (- np.sin(new_theta) * u1 + np.cos(new_theta) * u2) / (v + (dt / 2.) * k1[2])]

    new_theta = theta + (dt / 2.) * k2[3]
    u2 = omega * v * np.cos(new_theta)
    u1 = -u2 * np.tan(new_theta)

    k3 = [(v + (dt / 2.) * k2[2]) * np.cos(new_theta),
          (v + (dt / 2.) * k2[2]) * np.sin(new_theta),
          u1 * np.cos(new_theta) + u2 * np.sin(new_theta),
          (- np.sin(new_theta) * u1 + np.cos(new_theta) * u2) / (v + (dt / 2.) * k2[2])]

    new_theta = theta + dt * k3[3]
    u2 = omega * v * np.cos(new_theta)
    u1 = -u2 * np.tan(new_theta)

    k4 = [(v + dt * k3[2]) * np.cos(new_theta),
          (v + dt * k3[2]) * np.sin(new_theta),
          u1 * np.cos(new_theta) + u2 * np.sin(new_theta),
          (- np.sin(new_theta) * u1 + np.cos(new_theta) * u2) / (v + dt * k3[2])]

    new_x = [state[0] + 1.0 / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * dt,
             state[1] + 1.0 / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * dt,
             state[2] + 1.0 / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) * dt,
             state[3] + 1.0 / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) * dt]

    x_new = new_x[0]
    y_new = new_x[1]
    x_dot_new = new_x[2] * np.cos(new_x[3])
    y_dot_new = new_x[2] * np.sin(new_x[3])

    result = [x_new, y_new, x_dot_new, y_dot_new]

    return result


def unicycle_straight(xs):
    x = xs[0]
    y = xs[1]
    x_dot = xs[2]
    y_dot = xs[3]

    dt = 2.0

    x_new = x + dt*x_dot
    y_new = y + dt * y_dot
    x_dot_new = x_dot
    y_dot_new = y_dot

    result = [x_new, y_new, x_dot_new, y_dot_new]

    return result


def unicycle_slow(xs):
    x = xs[0]
    y = xs[1]
    x_dot = xs[2]
    y_dot = xs[3]

    dt = 2.0

    x_dot_new = x_dot/2.0
    y_dot_new = y_dot/2.0

    x_new = x + dt * (x_dot + x_dot_new)/2.0
    y_new = y + dt * (y_dot + y_dot_new)/2.0

    result = [x_new, y_new, x_dot_new, y_dot_new]

    return result


def unicycle_speed(xs):
    x = xs[0]
    y = xs[1]
    x_dot = xs[2]
    y_dot = xs[3]

    dt = 2.0

    x_dot_new = x_dot * 1.5
    y_dot_new = y_dot * 1.5

    x_new = x + dt * (x_dot + x_dot_new)/2.0
    y_new = y + dt * (y_dot + y_dot_new)/2.0

    result = [x_new, y_new, x_dot_new, y_dot_new]

    return result


def f(x, g, known_fnc, process_dist=None):
    if known_fnc is not None:
        x_known = known_fnc(x)
        output = np.add(x_known, g(x))
    else:
        output = g(x)

    n_dims_out = len(output)
    noise = [0] * n_dims_out
    if process_dist is not None:
        sig = process_dist["sig"]
        noise = np.random.uniform(-sig, sig, size=(n_dims_out,))

    return np.add(output, noise)


def known_part(x):
    return x


def generate_training_data(unknown_fnc, domain, data_num, known_fnc=None, random_seed=11, n_dims_out=-1,
                           process_dist=None, measurement_dist=None):
    # if known_fnc is None:
    f_sub = unknown_fnc
    # else:
    #     def f_sub(x):
    #         return np.add(known_fnc(x), unknown_fnc(x))

    n_dims_in = len(domain)
    if n_dims_out > 0:
        n_dims_out = n_dims_out
    else:
        n_dims_out = n_dims_in

    np.random.seed(random_seed)

    keys = list(domain)
    x_train = [np.random.uniform(domain[k][0], domain[k][1], data_num) for k in keys]
    x_train = np.reshape(x_train, [n_dims_in, data_num])

    y_train = [f_sub([x_train[idx_i][idx_j] for idx_i in range(n_dims_in)]) for idx_j in range(data_num)]
    y_train = np.transpose(np.reshape(y_train, [data_num, n_dims_out]))

    if process_dist is not None:
        # this is essentially the same as measurement noise for this method of generating data
        sig = process_dist["sig"]
        dist = "uniform"
        if "dist" in list(process_dist):
            dist = process_dist["dist"]
        if dist is "uniform":
            noise = np.random.uniform(-sig, sig, size=(n_dims_out, data_num))
        elif dist is "normal":
            noise = np.random.uniform(0., sig, size=(n_dims_out, data_num))
        else:
            error_print("Not an implemented noise distribution. Please check process distribution inputs.")
        y_train += noise

    if measurement_dist is not None:
        sig = measurement_dist["sig"]
        noise = np.random.uniform(-sig, sig, size=(n_dims_out, data_num))
        y_train += noise

    if known_fnc is not None:
        y_train -= np.transpose(
            np.reshape([known_fnc([x_train[idx_i][idx_j] for idx_i in range(n_dims_in)]) for idx_j in range(data_num)],
                       [data_num, n_dims_out]))

    # make sure x measurements are also corrupted by the noise
    # if measurement_dist is not None:
    #     sig = measurement_dist["sig"]
    #     noise = np.random.uniform(-sig, sig, size=(n_dims_out, data_num))
    #     x_train += noise

    assert np.size(x_train) == np.size(y_train)

    return x_train, y_train

