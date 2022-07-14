import os

def listfile_stage2_train(input_path, dataset_path, soil):
    train_12=[]
    train_13=[]
    train_14=[]
    train_23=[]
    train_24=[]
    train_34=[]
    train_12_conf=[]
    train_13_conf=[]
    train_14_conf=[]
    train_23_conf=[]
    train_24_conf=[]
    train_34_conf=[]
    train_rgb1=[]
    train_rgb2=[]
    train_rgb3=[]
    train_rgb4=[]
    train_gt = []

    val_12=[]
    val_13=[]
    val_14=[]
    val_23=[]
    val_24=[]
    val_34=[]
    val_12_conf=[]
    val_13_conf=[]
    val_14_conf=[]
    val_23_conf=[]
    val_24_conf=[]
    val_34_conf=[]
    val_rgb1=[]
    val_rgb2=[]
    val_rgb3=[]
    val_rgb4=[]
    val_gt = []

    ep_list = ["ep%d_500frames"%i for i in range(1,7)]
    for ep in ep_list:
        for subset in ['training', 'validation']:
            if soil:
                disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth_soiled")
                conf_map_path = os.path.join(input_path, ep, subset, "conf_map_soiled")
                rgb_path = os.path.join(dataset_path, ep, subset, "rgb_soiled")
            else:
                disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth")
                conf_map_path = os.path.join(input_path, ep, subset, "conf_map")
                rgb_path = os.path.join(dataset_path, ep, subset, "rgb")
            depth_path = os.path.join(dataset_path, ep, subset, "depth")
            
            input_name_list = os.listdir(disp_pred2depth_path)
            input_name_list.sort()
            conf_name_list = os.listdir(conf_map_path)
            conf_name_list.sort()
            rgb_name_list = os.listdir(rgb_path)
            rgb_name_list.sort()
            depth_name_list = os.listdir(depth_path)
            depth_name_list.sort()

            if subset == 'training':
                for frame in range(len(depth_name_list)):
                    train_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6]))
                    train_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+1]))
                    train_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+2]))
                    train_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+3]))
                    train_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+4]))
                    train_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+5]))
                    train_12_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6]))
                    train_13_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+1]))
                    train_14_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+2]))
                    train_23_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+3]))
                    train_24_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+4]))
                    train_34_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+5]))
                    train_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame*12]))
                    train_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame*12+1]))
                    train_rgb3.append(os.path.join(rgb_path, rgb_name_list[frame*12+10]))
                    train_rgb4.append(os.path.join(rgb_path, rgb_name_list[frame*12+11]))
                    train_gt.append(os.path.join(depth_path, depth_name_list[frame]))
            else:
                for frame in range(len(depth_name_list)):
                    val_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6]))
                    val_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+1]))
                    val_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+2]))
                    val_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+3]))
                    val_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+4]))
                    val_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+5]))
                    val_12_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6]))
                    val_13_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+1]))
                    val_14_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+2]))
                    val_23_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+3]))
                    val_24_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+4]))
                    val_34_conf.append(os.path.join(conf_map_path, conf_name_list[frame*6+5]))
                    val_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame*12]))
                    val_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame*12+1]))
                    val_rgb3.append(os.path.join(rgb_path, rgb_name_list[frame*12+10]))
                    val_rgb4.append(os.path.join(rgb_path, rgb_name_list[frame*12+11]))
                    val_gt.append(os.path.join(depth_path, depth_name_list[frame]))

    train_depthes = [train_12, train_13, train_14, train_23, train_24, train_34]
    train_confs = [train_12_conf, train_13_conf, train_14_conf, train_23_conf, train_24_conf, train_34_conf]
    train_rgbs = [train_rgb1, train_rgb2, train_rgb3, train_rgb4]
    val_depthes = [val_12, val_13, val_14, val_23, val_24, val_34]
    val_confs = [val_12_conf, val_13_conf, val_14_conf, val_23_conf, val_24_conf, val_34_conf]
    val_rgbs = [val_rgb1, val_rgb2, val_rgb3, val_rgb4]
    return train_depthes, train_confs, train_rgbs, train_gt, val_depthes, val_confs, val_rgbs, val_gt

def listfile_stage2_test(input_path, dataset_path, soil):
    test_12=[]
    test_13=[]
    test_14=[]
    test_23=[]
    test_24=[]
    test_34=[]
    test_12_conf=[]
    test_13_conf=[]
    test_14_conf=[]
    test_23_conf=[]
    test_24_conf=[]
    test_34_conf=[]
    test_rgb1=[]
    test_rgb2=[]
    test_rgb3=[]
    test_rgb4=[]
    test_gt = []

    ep_list = ["ep%d_500frames"%i for i in range(1,7)]
    subset = "testing"
    for ep in ep_list:
        if soil:
            disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth_soiled")
            conf_path = os.path.join(input_path, ep, subset, "conf_map_soiled")
            rgb_path = os.path.join(dataset_path, ep, subset, "rgb_soiled")
        else:
            disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth")
            conf_path = os.path.join(input_path, ep, subset, "conf_map")
            rgb_path = os.path.join(dataset_path, ep, subset, "rgb")
        depth_path = os.path.join(dataset_path, ep, subset, "depth")

        input_name_list = os.listdir(disp_pred2depth_path)
        input_name_list.sort()
        conf_name_list = os.listdir(conf_path)
        conf_name_list.sort()
        rgb_name_list = os.listdir(rgb_path)
        rgb_name_list.sort()
        depth_name_list = os.listdir(depth_path)
        depth_name_list.sort()
        
        for frame in range(len(depth_name_list)):
            test_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6]))
            test_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+1]))
            test_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+2]))
            test_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+3]))
            test_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+4]))
            test_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+5]))
            test_12_conf.append(os.path.join(conf_path, conf_name_list[frame*6]))
            test_13_conf.append(os.path.join(conf_path, conf_name_list[frame*6+1]))
            test_14_conf.append(os.path.join(conf_path, conf_name_list[frame*6+2]))
            test_23_conf.append(os.path.join(conf_path, conf_name_list[frame*6+3]))
            test_24_conf.append(os.path.join(conf_path, conf_name_list[frame*6+4]))
            test_34_conf.append(os.path.join(conf_path, conf_name_list[frame*6+5]))
            test_gt.append(os.path.join(depth_path, depth_name_list[frame]))
            test_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame*12]))
            test_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame*12+1]))
            test_rgb3.append(os.path.join(rgb_path, rgb_name_list[frame*12+10]))
            test_rgb4.append(os.path.join(rgb_path, rgb_name_list[frame*12+11]))

    test_depthes = [test_12, test_13, test_14, test_23, test_24, test_34]
    test_confs = [test_12_conf, test_13_conf, test_14_conf, test_23_conf, test_24_conf, test_34_conf]
    test_rgbs = [test_rgb1, test_rgb2, test_rgb3, test_rgb4]
    return test_depthes, test_confs, test_rgbs, test_gt