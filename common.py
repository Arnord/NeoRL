import matplotlib.pyplot as plt


def get_traj_plt(v):
    x_prefix = range(len(v['state']))
    y_h_star_list = []
    y_h_list = []
    y_c_star_list = []
    y_c_list = []
    u_fu_list = []
    u_ff_list = []
    for i in range(len(v['state'])):    # [[state]], [action]
        y_h_star_list.append(v['state'][i][0][0])
        y_c_star_list.append(v['state'][i][0][1])
        y_h_list.append(v['state'][i][0][2])
        y_c_list.append(v['state'][i][0][3])
        u_fu_list.append(v['action'][i][0])
        u_ff_list.append(v['action'][i][1])

    fig = plt.figure()
    plt.subplot(411)
    plt.plot(x_prefix, y_h_list, label='y_h')
    plt.plot(x_prefix, y_h_star_list, label='y_h_star')
    plt.legend()

    plt.subplot(412)
    plt.plot(x_prefix, y_c_list, label='y_c')
    plt.plot(x_prefix, y_c_star_list, label='y_c_star')
    plt.legend()

    plt.subplot(413)
    plt.plot(x_prefix, u_fu_list, label='u_fu')
    plt.legend()

    plt.subplot(414)
    plt.plot(x_prefix, u_ff_list, label='u_ff')
    plt.legend()

    plt.close(fig)

    return fig

