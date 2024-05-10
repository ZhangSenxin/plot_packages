from zsx_pie_plot_beta2 import *


def sector_data(family):
    k = len(family)
    n = sum(family)

    Alpha = np.array([(family[i] + 1) * 2 * pi / (np.sum(family) + k) for i in range(k)])
    x_angle0 = [0] + [sum(Alpha[:(i + 1)]) for i in range(k)]
    x_angle0[-1] = 2 * pi
    k0 = [math.tan(i) for i in x_angle0]

    x_alpha = []
    for i in range(k):
        alpha = [Alpha[i] / (family[i] + 1)] * (family[i])
        if i > 0:
            alpha[0] += Alpha[i - 1] / (family[i - 1] + 1)
        x_alpha += alpha
    x_angle = np.array([sum(x_alpha[:(i + 1)]) for i in range(n)])

    return x_angle0, k0, x_angle, k


def circos_old(x_angle, bar_data, i, color, interval, zorder=10):
    angle = x_angle[i]
    if angle < 0.5 * pi:
        x = math.cos(angle)
        # y = math.sin(angle)
        x_t = math.sin(angle) * interval
        # y_t = - math.cos(angle) * interval

        if x + x_t < x * (1 + bar_data[i]) - x_t:
            x_plot_f = np.arange(x - x_t, x + x_t, 0.001)
            x_plot_m = np.arange(x + x_t, x * (1 + bar_data[i]) - x_t, 0.001)
            x_plot_b = np.arange(x * (1 + bar_data[i]) - x_t, x * (1 + bar_data[i]) + x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle) + interval / math.cos(angle)
            y_plot_f_d = x_plot_f * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_m_u = x_plot_m * math.tan(angle) + interval / math.cos(angle)
            y_plot_m_d = x_plot_m * math.tan(angle) - interval / math.cos(angle)
            y_plot_b_u = x_plot_b * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_b_d = x_plot_b * math.tan(angle) - interval / math.cos(angle)

        else:
            x_plot_f = np.arange(x - x_t, x * (1 + bar_data[i]) - x_t, 0.001)
            x_plot_m = np.arange(x * (1 + bar_data[i]) - x_t, x + x_t, 0.001)
            x_plot_b = np.arange(x + x_t, x * (1 + bar_data[i]) + x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle) + interval / math.cos(angle)
            y_plot_f_d = x_plot_f * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_m_u = x_plot_m * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_m_d = x_plot_m * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_b_u = x_plot_b * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_b_d = x_plot_b * math.tan(angle) - interval / math.cos(angle)

    elif angle == 0.5 * pi:
        x = math.cos(angle)
        # y = math.sin(angle)
        x_t = math.sin(angle) * interval
        # y_t = - math.cos(angle) * interval

        x_plot_f = np.arange(x - x_t, x + x_t, 0.001)
        x_plot_m = np.arange(0, 0, 0.001)
        x_plot_b = np.arange(0, 0, 0.001)

        y_plot_f_u = 1 + bar_data[i]
        y_plot_f_d = 1

        y_plot_m_u = x_plot_m * math.tan(angle) + interval / math.cos(angle)
        y_plot_m_d = x_plot_m * math.tan(angle) - interval / math.cos(angle)
        y_plot_b_u = x_plot_b * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
        y_plot_b_d = x_plot_b * math.tan(angle) - interval / math.cos(angle)

    elif 0.5 * pi < angle <= pi:
        x = math.cos(angle)
        # y = math.sin(angle)
        x_t = math.sin(angle) * interval
        # y_t = - math.cos(angle) * interval

        if x - x_t > x * (1 + bar_data[i]) + x_t:
            x_plot_f = np.arange(x * (1 + bar_data[i]) - x_t, x * (1 + bar_data[i]) + x_t, 0.001)
            x_plot_m = np.arange(x * (1 + bar_data[i]) + x_t, x - x_t, 0.001)
            x_plot_b = np.arange(x - x_t, x + x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_f_d = x_plot_f * math.tan(angle) + interval / math.cos(angle)
            y_plot_m_u = x_plot_m * math.tan(angle) - interval / math.cos(angle)
            y_plot_m_d = x_plot_m * math.tan(angle) + interval / math.cos(angle)
            y_plot_b_u = x_plot_b * math.tan(angle) - interval / math.cos(angle)
            y_plot_b_d = x_plot_b * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)

        else:
            x_plot_f = np.arange(x * (1 + bar_data[i]) - x_t, x - x_t, 0.001)
            x_plot_m = np.arange(x - x_t, x * (1 + bar_data[i]) + x_t, 0.001)
            x_plot_b = np.arange(x * (1 + bar_data[i]) + x_t, x + x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_f_d = x_plot_f * math.tan(angle) + interval / math.cos(angle)
            y_plot_m_u = x_plot_m * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_m_d = x_plot_m * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_b_u = x_plot_b * math.tan(angle) - interval / math.cos(angle)
            y_plot_b_d = x_plot_b * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)

    elif pi < angle < 1.5 * pi:
        x = math.cos(angle)
        # y = math.sin(angle)
        x_t = math.sin(angle) * interval
        # y_t = - math.cos(angle) * interval

        if x + x_t > x * (1 + bar_data[i]) - x_t:
            x_plot_f = np.arange(x * (1 + bar_data[i]) + x_t, x * (1 + bar_data[i]) - x_t, 0.001)
            x_plot_m = np.arange(x * (1 + bar_data[i]) - x_t, x + x_t, 0.001)
            x_plot_b = np.arange(x + x_t, x - x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle) - interval / math.cos(angle)
            y_plot_f_d = x_plot_f * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_m_u = x_plot_m * math.tan(angle) - interval / math.cos(angle)
            y_plot_m_d = x_plot_m * math.tan(angle) + interval / math.cos(angle)
            y_plot_b_u = x_plot_b * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_b_d = x_plot_b * math.tan(angle) + interval / math.cos(angle)

        else:
            x_plot_f = np.arange(x * (1 + bar_data[i]) + x_t, x + x_t, 0.001)
            x_plot_m = np.arange(x + x_t, x * (1 + bar_data[i]) - x_t, 0.001)
            x_plot_b = np.arange(x * (1 + bar_data[i]) - x_t, x - x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle) - interval / math.cos(angle)
            y_plot_f_d = x_plot_f * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_m_u = x_plot_m * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_m_d = x_plot_m * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_b_u = x_plot_b * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_b_d = x_plot_b * math.tan(angle) + interval / math.cos(angle)

    elif angle == 1.5 * pi:
        x = math.cos(angle)
        # y = math.sin(angle)
        x_t = math.sin(angle) * interval
        # y_t = - math.cos(angle) * interval

        x_plot_f = np.arange(x + x_t, x - x_t, 0.001)
        x_plot_m = np.arange(0, 0, 0.001)
        x_plot_b = np.arange(0, 0, 0.001)

        y_plot_f_u = -1
        y_plot_f_d = -1 - bar_data[i]

        y_plot_m_u = x_plot_m * math.tan(angle) + interval / math.cos(angle)
        y_plot_m_d = x_plot_m * math.tan(angle) - interval / math.cos(angle)
        y_plot_b_u = x_plot_b * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
        y_plot_b_d = x_plot_b * math.tan(angle) - interval / math.cos(angle)

    elif angle > 1.5 * pi:
        x = math.cos(angle)
        # y = math.sin(angle)
        x_t = math.sin(angle) * interval
        # y_t = - math.cos(angle) * interval

        if x * (1 + bar_data[i]) + x_t > x - x_t:
            x_plot_f = np.arange(x + x_t, x - x_t, 0.001)
            x_plot_m = np.arange(x - x_t, x * (1 + bar_data[i]) + x_t, 0.001)
            x_plot_b = np.arange(x * (1 + bar_data[i]) + x_t, x * (1 + bar_data[i]) - x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_f_d = x_plot_f * math.tan(angle) - interval / math.cos(angle)
            y_plot_m_u = x_plot_m * math.tan(angle) + interval / math.cos(angle)
            y_plot_m_d = x_plot_m * math.tan(angle) - interval / math.cos(angle)
            y_plot_b_u = x_plot_b * math.tan(angle) + interval / math.cos(angle)
            y_plot_b_d = x_plot_b * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)

        else:
            x_plot_f = np.arange(x + x_t, x * (1 + bar_data[i]) + x_t, 0.001)
            x_plot_m = np.arange(x * (1 + bar_data[i]) + x_t, x - x_t, 0.001)
            x_plot_b = np.arange(x - x_t, x * (1 + bar_data[i]) - x_t, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_f_d = x_plot_f * math.tan(angle) - interval / math.cos(angle)
            y_plot_m_u = x_plot_m * math.tan(angle + 0.5 * pi) + 1 / math.sin(angle)
            y_plot_m_d = x_plot_m * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)
            y_plot_b_u = x_plot_b * math.tan(angle) + interval / math.cos(angle)
            y_plot_b_d = x_plot_b * math.tan(angle + 0.5 * pi) + (1 + bar_data[i]) / math.tan(angle) / math.cos(angle)

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot, y_plot_u, y_plot_d, where=(y_plot_u > y_plot_d), color=color, zorder=zorder)


# bar to be a triangle
def circos(x_angle, bar_data, i, interval, color=False, zorder=10, edgeprop=False):
    angle = x_angle[i]
    angle1 = angle-interval
    angle2 = angle+interval
    x10 = math.cos(angle1)
    y10 = math.sin(angle1)
    x20 = math.cos(angle2)
    y20 = math.sin(angle2)
    x1 = math.cos(angle1) * (1 + bar_data[i])
    y1 = math.sin(angle1) * (1 + bar_data[i])
    x2 = math.cos(angle2) * (1 + bar_data[i])
    y2 = math.sin(angle2) * (1 + bar_data[i])

    if not color:
        color = ['lightpink']*999

    if not edgeprop:
        edgeprop = {'lw': 1, 'color': 'lightgray'}
    else:
        try:
            edgeprop['lw']
        except KeyError:
            edgeprop['lw'] = 1
        try:
            edgeprop['color']
        except KeyError:
            edgeprop['color'] = 'lightgray'

        plt.plot([x2, x20], [y2, y20], lw=edgeprop['lw'], color=edgeprop['color'])
        plt.plot([x1, x2], [y1, y2], lw=edgeprop['lw'], color=edgeprop['color'])
        plt.plot([x1, x10], [y1, y10], lw=edgeprop['lw'], color=edgeprop['color'])

    if angle2 <= 0.5 * pi:
        if x10 < x2:
            x_plot_f = np.arange(x20, x10, 0.001)
            x_plot_m = np.arange(x10, x2, 0.001)
            x_plot_b = np.arange(x2, x1, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle2)
            y_plot_f_d = x_plot_f * ((y20-y10)/(x20-x10)) + (y10*x20-x10*y20)/(x20-x10)
            y_plot_m_u = x_plot_m * math.tan(angle2)
            y_plot_m_d = x_plot_m * math.tan(angle1)
            y_plot_b_u = x_plot_b * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
            y_plot_b_d = x_plot_b * math.tan(angle1)

        else:
            x_plot_f = np.arange(x20, x2, 0.001)
            x_plot_m = np.arange(x2, x10, 0.001)
            x_plot_b = np.arange(x10, x1, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle2)
            y_plot_f_d = x_plot_f * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)
            y_plot_m_u = x_plot_m * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)
            y_plot_m_d = x_plot_m * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)
            y_plot_b_u = x_plot_b * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)
            y_plot_b_d = x_plot_b * math.tan(angle1)

    elif (angle2 > 0.5 * pi) and (angle1 < 0.5 * pi):
        x_plot_f = np.arange(x2, x20, 0.001)
        x_plot_m = np.arange(x20, x10, 0.001)
        x_plot_b = np.arange(x10, x1, 0.001)

        y_plot_f_u = x_plot_f * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
        y_plot_f_d = x_plot_f * math.tan(angle2)
        y_plot_m_u = x_plot_m * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
        y_plot_m_d = x_plot_m * ((y20-y10)/(x20-x10)) + (y10*x20-x10*y20)/(x20-x10)
        y_plot_b_u = x_plot_b * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
        y_plot_b_d = x_plot_b * math.tan(angle1)

    elif 0.5 * pi < angle1 and (angle < pi):
        if x1 <= x20:
            x_plot_f = np.arange(x2, x1, 0.001)
            x_plot_m = np.arange(x1, x20, 0.001)
            x_plot_b = np.arange(x20, x10, 0.001)

            y_plot_f_u = x_plot_f * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
            y_plot_f_d = x_plot_f * math.tan(angle2)
            y_plot_m_u = x_plot_m * math.tan(angle1)
            y_plot_m_d = x_plot_m * math.tan(angle2)
            y_plot_b_u = x_plot_b * math.tan(angle1)
            y_plot_b_d = x_plot_b * ((y20-y10)/(x20-x10)) + (y10*x20-x10*y20)/(x20-x10)

        else:
            x_plot_f = np.arange(x2, x20, 0.001)
            x_plot_m = np.arange(x20, x1, 0.001)
            x_plot_b = np.arange(x1, x10, 0.001)

            y_plot_f_u = x_plot_f * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)
            y_plot_f_d = x_plot_f * math.tan(angle2)
            y_plot_m_u = x_plot_m * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)
            y_plot_m_d = x_plot_m * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)
            y_plot_b_u = x_plot_b * math.tan(angle1)
            y_plot_b_d = x_plot_b * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)

    elif angle == pi:
        x_plot_f = np.arange(x1, x1, 0.001)
        x_plot_m = np.arange(x1, x20, 0.001)
        x_plot_b = np.arange(x10, x10, 0.001)

        y_plot_f_u = x_plot_f * 0
        y_plot_f_d = x_plot_f * 0
        y_plot_m_u = x_plot_m * math.tan(angle1)
        y_plot_m_d = x_plot_m * math.tan(angle2)
        y_plot_b_u = x_plot_b * 0
        y_plot_b_d = x_plot_b * 0

    elif (pi < angle) and (angle2 <= 1.5 * pi):
        if x2 < x10:
            x_plot_f = np.arange(x1, x2, 0.001)
            x_plot_m = np.arange(x2, x10, 0.001)
            x_plot_b = np.arange(x10, x20, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle1)
            y_plot_f_d = x_plot_f * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
            y_plot_m_u = x_plot_m * math.tan(angle1)
            y_plot_m_d = x_plot_m * math.tan(angle2)
            y_plot_b_u = x_plot_b * ((y20-y10)/(x20-x10)) + (y10*x20-x10*y20)/(x20-x10)
            y_plot_b_d = x_plot_b * math.tan(angle2)

        else:
            x_plot_f = np.arange(x1, x10, 0.001)
            x_plot_m = np.arange(x10, x2, 0.001)
            x_plot_b = np.arange(x2, x20, 0.001)

            y_plot_f_u = x_plot_f * math.tan(angle1)
            y_plot_f_d = x_plot_f * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)
            y_plot_m_u = x_plot_m * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)
            y_plot_m_d = x_plot_m * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)
            y_plot_b_u = x_plot_b * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)
            y_plot_b_d = x_plot_b * math.tan(angle2)

    elif (angle2 > 1.5 * pi) and (angle1 < 1.5 * pi):
        x_plot_f = np.arange(x1, x10, 0.001)
        x_plot_m = np.arange(x10, x20, 0.001)
        x_plot_b = np.arange(x20, x2, 0.001)

        y_plot_f_u = x_plot_f * math.tan(angle1)
        y_plot_f_d = x_plot_f * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
        y_plot_m_u = x_plot_m * ((y20-y10)/(x20-x10)) + (y10*x20-x10*y20)/(x20-x10)
        y_plot_m_d = x_plot_m * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)
        y_plot_b_u = x_plot_b * math.tan(angle2)
        y_plot_b_d = x_plot_b * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)

    else:
        if x20 < x1:
            x_plot_f = np.arange(x10, x20, 0.001)
            x_plot_m = np.arange(x20, x1, 0.001)
            x_plot_b = np.arange(x1, x2, 0.001)

            y_plot_f_u = x_plot_f * ((y20-y10)/(x20-x10)) + (y10*x20-x10*y20)/(x20-x10)
            y_plot_f_d = x_plot_f * math.tan(angle1)
            y_plot_m_u = x_plot_m * math.tan(angle2)
            y_plot_m_d = x_plot_m * math.tan(angle1)
            y_plot_b_u = x_plot_b * math.tan(angle2)
            y_plot_b_d = x_plot_b * ((y2-y1)/(x2-x1)) + (y1*x2-x1*y2)/(x2-x1)

        else:
            x_plot_f = np.arange(x10, x1, 0.001)
            x_plot_m = np.arange(x1, x20, 0.001)
            x_plot_b = np.arange(x20, x2, 0.001)

            y_plot_f_u = x_plot_f * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)
            y_plot_f_d = x_plot_f * math.tan(angle1)
            y_plot_m_u = x_plot_m * ((y20 - y10) / (x20 - x10)) + (y10 * x20 - x10 * y20) / (x20 - x10)
            y_plot_m_d = x_plot_m * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)
            y_plot_b_u = x_plot_b * math.tan(angle2)
            y_plot_b_d = x_plot_b * ((y2 - y1) / (x2 - x1)) + (y1 * x2 - x1 * y2) / (x2 - x1)

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot, y_plot_u, y_plot_d, where=(y_plot_u > y_plot_d), color=color, zorder=zorder)


def marker_V(x_angle, i, color_list=False, r=0.7, poly=3, style=0, rotation=90, s=200, zorder=21, ec=False):
    angle = x_angle[i]
    x = math.cos(angle) * r
    y = math.sin(angle) * r

    if not color_list:
        color_list = ['red'] * 999

    if ec:
        plt.scatter(x, y, c=color_list[i], s=s, edgecolors=ec[i], marker=(poly, style, angle * 180 / pi - rotation),
                    zorder=zorder)
    else:
        plt.scatter(x, y, c=color_list[i], s=s, marker=(poly, style, angle * 180 / pi - rotation), zorder=zorder)


def label_V(x_angle, i, label, color_list=False, r=0.85, rotation=90, fs=12, zorder=21):
    angle = 0.5 * (x_angle[i-1]+x_angle[i])
    x = math.cos(angle)*r
    y = math.sin(angle)*r

    if not color_list:
        color_list = ['black'] * 999

    plt.text(x, y, label[i], fontsize=fs, rotation=angle*180/pi-rotation, color=color_list[i],
             verticalalignment="center", horizontalalignment="center", zorder=zorder)


def scatter_V(x_angle, scatter_data, i, color_list=False, s=5, zorder=21):
    angle = x_angle[i]
    if type(scatter_data) == list:
        x = [math.cos(angle) *  (1 + scatter_data[i][j]) for j in range(len(scatter_data[i]))]
        y = [math.sin(angle) *  (1 + scatter_data[i][j]) for j in range(len(scatter_data[i]))]
    elif type(scatter_data) == np.ndarray:
        x = [math.cos(angle) *  (1 + scatter_data[i, j]) for j in range(len(scatter_data[i]))]
        y = [math.sin(angle) *  (1 + scatter_data[i, j]) for j in range(len(scatter_data[i]))]

    if not color_list:
        color = ['black'] * len(x)
    else:
        color = [color_list[i]] * len(x)

    plt.scatter(x, y, c=color, s=s, zorder=zorder)
