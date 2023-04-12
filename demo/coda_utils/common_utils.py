

def drop_info_with_name(info, name, gt_filtered=False):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        if gt_filtered and (key=='gt_boxes_lidar' or key=='index'):
            ret_info[key] = info[key]
            continue #already filtered non gt classes in coda
        ret_info[key] = info[key][keep_indices]
    return ret_info