def manual_actions_fn(gestures, gripper_open):
    p__an = []
    for a in range(n_actions): # row is selection by action
        p__an_o = np.dot(gestures, cptm1[a]) * \
                  np.dot(gripper_open, mapping_table_gr2a[a])
        p__an.append(p__an_o)

    return p__an
