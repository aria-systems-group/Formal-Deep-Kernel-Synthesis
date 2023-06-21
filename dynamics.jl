
using Random
using Distributions

function sys_2d_dynamics(variance)
    trust_1 = false
    if trust_1
        mode1 = (x) -> [x[1] + 0.25 + 0.05*sin(x[2]),
                        x[2] + 0.1*cos(x[1])]

        mode2 = (x) -> [x[1] - 0.25 + 0.05*sin(x[2]),
                        x[2] + 0.1*cos(x[1])]

        mode3 = (x) -> [x[1] + 0.1*cos(x[2]),
                        x[2] + 0.25 + 0.05*sin(x[1])]

        mode4 = (x) -> [x[1] + 0.1*cos(x[2]),
                        x[2] - 0.25 + 0.05*sin(x[1])]

    else
        mode1 = (x) -> [x[1] + 0.5 + 0.2*sin(x[2]),
                        x[2] + 0.4*cos(x[1])]

        mode2 = (x) -> [x[1] - 0.5 + 0.2*sin(x[2]),
                        x[2] + 0.4*cos(x[1])]

        mode3 = (x) -> [x[1] + 0.4*cos(x[2]),
                        x[2] + 0.5 + 0.2*sin(x[1])]

        mode4 = (x) -> [x[1] + 0.4*cos(x[2]),
                        x[2] - 0.5 + 0.2*sin(x[1])]
    end

    process_noise = Normal(0., sqrt(variance))
    f1 = (x) -> mode1(x) + rand(process_noise, (2,1))
    f2 = (x) -> mode2(x) + rand(process_noise, (2,1))
    f3 = (x) -> mode3(x) + rand(process_noise, (2,1))
    f4 = (x) -> mode4(x) + rand(process_noise, (2,1))

    return [f1, f2, f3, f4]
end


function sys_2d_as_3d_dynamics(variance)

    mode1 = (x) -> [x[1] + 0.5 + 0.2*sin(x[2]),
                    x[2] + 0.4*cos(x[1]),
                    0.75*x[3] + 0.1*cos(x[1])]

    mode2 = (x) -> [x[1] - 0.5 + 0.2*sin(x[2]),
                    x[2] + 0.4*cos(x[1]),
                    0.75*x[3] + 0.1*cos(x[1])]

    mode3 = (x) -> [x[1] + 0.4*cos(x[2]),
                    x[2] + 0.5 + 0.2*sin(x[1]),
                    0.75*x[3] + 0.1*cos(x[1])]

    mode4 = (x) -> [x[1] + 0.4*cos(x[2]),
                    x[2] - 0.5 + 0.2*sin(x[1]),
                    0.75*x[3] + 0.1*cos(x[1])]

    process_noise = Normal(0., sqrt(variance))
    f1 = (x) -> mode1(x) + rand(process_noise, (3,1))
    f2 = (x) -> mode2(x) + rand(process_noise, (3,1))
    f3 = (x) -> mode3(x) + rand(process_noise, (3,1))
    f4 = (x) -> mode4(x) + rand(process_noise, (3,1))

    return [f1, f2, f3, f4]
end

function sys_3d_dynamics(variance)
    u = 10
    omega = 5
    Ts = 0.1

    phi = -0.3
    mode1 = (x) -> [x[1] + Ts * u * cos(x[3]),
                    x[2] + Ts * u * sin(x[3]),
                    x[3] + (phi - x[3]) * Ts * omega]


    phi = -0.15
    mode2 = (x) -> [x[1] + Ts * u * cos(x[3]),
                    x[2] + Ts * u * sin(x[3]),
                    x[3] + (phi - x[3]) * Ts * omega]


    phi = 0.
    mode3 = (x) -> [x[1] + Ts * u * cos(x[3]),
                    x[2] + Ts * u * sin(x[3]),
                    x[3] + (phi - x[3]) * Ts * omega]

    phi = 0.3
    mode4 = (x) -> [x[1] + Ts * u * cos(x[3]),
                    x[2] + Ts * u * sin(x[3]),
                    x[3] + (phi - x[3]) * Ts * omega]

    phi = 0.15
    mode5 = (x) -> [x[1] + Ts * u * cos(x[3]),
                    x[2] + Ts * u * sin(x[3]),
                    x[3] + (phi - x[3]) * Ts * omega]

    phi = 0.45
    mode6 = (x) -> [x[1] + Ts * u * cos(x[3]),
                    x[2] + Ts * u * sin(x[3]),
                    x[3] + (phi - x[3]) * Ts * omega]


    phi = -0.45
    mode7 = (x) -> [x[1] + Ts * u * cos(x[3]),
                    x[2] + Ts * u * sin(x[3]),
                    x[3] + (phi - x[3]) * Ts * omega]


    noise_x = Normal(0., sqrt(variance[1]))
    noise_theta = Normal(0., sqrt(variance[3]))
    f1 = (x) -> mode1(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1]]
    f2 = (x) -> mode2(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1]]
    f3 = (x) -> mode3(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1]]
    f4 = (x) -> mode4(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1]]
    f5 = (x) -> mode5(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1]]
    f6 = (x) -> mode6(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1]]
    f7 = (x) -> mode7(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1]]

    return [f1, f2, f3, f4, f5, f6, f7]
end


function sys_5d(variance)

    mode1 = (x) -> [x[1] + 0.35 + (0.1 * sin(x[2])),
                    x[2] + (0.15 * cos(x[1])) + (0.05 * x[3]),
                    (0.3 * x[3]) + (0.4 * x[4]),
                    (0.4 * x[4]) + (0.05 * x[5]),
                    (0.5 * x[5])]

    mode2 = (x) -> [x[1] - 0.35 + (0.1 * sin(x[2])),
                    x[2] + (0.15 * cos(x[1])) + (0.05 * x[3]),
                    (0.3 * x[3]) + (0.4 * x[4]),
                    (0.4 * x[4]) + (0.05 * x[5]),
                    (0.5 * x[5])]

    mode3 = (x) -> [x[1] + (0.15 * cos(x[2])),
                    x[2] + 0.35 + (0.1 * sin(x[1])) + (0.05 * x[3]),
                    (0.3 * x[3]) + (0.4 * x[4]),
                    (0.4 * x[4]) + (0.05 * x[5]),
                    (0.5 * x[5])]

    mode4 = (x) -> [x[1] + (0.15 * cos(x[2])),
                    x[2] - 0.35 + (0.1 * sin(x[1])) + (0.05 * x[3]),
                    (0.3 * x[3]) + (0.4 * x[4]),
                    (0.4 * x[4]) + (0.05 * x[5]),
                    (0.5 * x[5])]

    noise_x = Normal(0., sqrt(variance[1]))
    noise_theta = Normal(0., sqrt(variance[3]))
    f1 = (x) -> mode1(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1]]
    f2 = (x) -> mode2(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1]]
    f3 = (x) -> mode3(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1]]
    f4 = (x) -> mode4(x) + [rand(noise_x, 1)[1], rand(noise_x, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1], rand(noise_theta, 1)[1]]

    return [f1, f2, f3, f4]
end