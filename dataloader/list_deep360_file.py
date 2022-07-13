import os
import cv2

def listfile_stage1_train(filepath):

    train_left_img=[]
    train_right_img=[]
    train_left_disp = []
    val_left_img=[]
    val_right_img=[]
    val_left_disp = []

    ep_list = os.listdir(filepath)      #ep子文件夹名列表
    ep_list.sort()

    for ep in ep_list:
        for subset in ['training', 'validation']:
            rgb_path = os.path.join(filepath, ep, subset, "rgb")
            disp_path = os.path.join(filepath, ep, subset, "disp")
            
            rgb_name_list = os.listdir(rgb_path)
            rgb_name_list.sort()
            disp_name_list = os.listdir(disp_path)
            disp_name_list.sort()
        
            if subset == 'training':
                for i in range(len(disp_name_list)):
                    train_left_img.append(os.path.join(rgb_path, rgb_name_list[i*2]))
                    train_right_img.append(os.path.join(rgb_path, rgb_name_list[i*2+1]))
                    train_left_disp.append(os.path.join(disp_path, disp_name_list[i]))
            else:
                for i in range(len(disp_name_list)):
                    val_left_img.append(os.path.join(rgb_path, rgb_name_list[i*2]))
                    val_right_img.append(os.path.join(rgb_path, rgb_name_list[i*2+1]))
                    val_left_disp.append(os.path.join(disp_path, disp_name_list[i]))

    return train_left_img, train_right_img, train_left_disp, val_left_img, val_right_img, val_left_disp

def listfile_stage1_test(filepath):

    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    ep_list = os.listdir(filepath)      #ep子文件夹名列表
    ep_list.sort()

    subset = "testing"
    for ep in ep_list:
        rgb_path = os.path.join(filepath, ep, subset, "rgb")
        disp_path = os.path.join(filepath, ep, subset, "disp")
        
        rgb_name_list = os.listdir(rgb_path)
        rgb_name_list.sort()
        disp_name_list = os.listdir(disp_path)
        disp_name_list.sort()
    
        for i in range(len(disp_name_list)):
            test_left_img.append(os.path.join(rgb_path, rgb_name_list[i*2]))
            test_right_img.append(os.path.join(rgb_path, rgb_name_list[i*2+1]))
            test_left_disp.append(os.path.join(disp_path, disp_name_list[i]))

    return test_left_img, test_right_img, test_left_disp

def listfile_stage1_output(filepath, soil):

    left_img=[]
    right_img=[]

    ep_list = os.listdir(filepath)      #ep子文件夹名列表
    ep_list.sort()

    for ep in ep_list:
        for subset in ['training', 'validation', 'testing']:
            if not soil:
                rgb_path = os.path.join(filepath, ep, subset, "rgb")
                rgb_name_list = os.listdir(rgb_path)
                rgb_name_list.sort()

                for i in range(int(len(rgb_name_list)/2)):
                    left_img.append(os.path.join(rgb_path, rgb_name_list[i*2]))
                    right_img.append(os.path.join(rgb_path, rgb_name_list[i*2+1]))
            else:
                rgb_soiled_path = os.path.join(filepath, ep, subset, "rgb_soiled")
                if subset == 'testing':
                    soil_type_list = os.listdir(rgb_soiled_path)
                    for soil_type in soil_type_list:
                        soil_type_path = os.path.join(rgb_soiled_path, soil_type)
                        soil_cam_num_list = os.listdir(soil_type_path)
                        for soil_cam_num in soil_cam_num_list:
                            soil_cam_num_path = os.path.join(soil_type_path, soil_cam_num)
                            soil_num_list = os.listdir(soil_cam_num_path)
                            for soil_num in soil_num_list:
                                soil_num_path = os.path.join(soil_cam_num_path, soil_num)
                                soil_rate_list = os.listdir(soil_num_path)
                                for soil_rate in soil_rate_list:
                                    soil_rate_path = os.path.join(soil_num_path, soil_rate)
                                    rgb_soiled_name_list = os.listdir(soil_rate_path)
                                    rgb_soiled_name_list.sort()
                                    for i in range(int(len(rgb_soiled_name_list)/2)):
                                        left_img.append(os.path.join(soil_rate_path, rgb_soiled_name_list[i*2]))
                                        right_img.append(os.path.join(soil_rate_path, rgb_soiled_name_list[i*2+1]))
                else:
                    rgb_soiled_name_list = os.listdir(rgb_soiled_path)
                    rgb_soiled_name_list.sort()
                    for i in range(int(len(rgb_soiled_name_list)/2)):
                        left_img.append(os.path.join(rgb_soiled_path, rgb_soiled_name_list[i*2]))
                        right_img.append(os.path.join(rgb_soiled_path, rgb_soiled_name_list[i*2+1]))
            
    return left_img, right_img

def listfile_stage1_output_omnifisheye(filepath):

    left_img=[]
    right_img=[]
    masks = []

    mask_name_list = os.listdir(filepath)
    mask_name_list.remove('training')
    mask_name_list.remove('testing')
    mask_name_list.sort()
    for mask_name in mask_name_list:
        mask = cv2.imread(filepath+mask_name)
        masks.append(1.0 - mask[:,:,0]/255.0)

    for subset in ['training', 'testing']:
        rgb_path = os.path.join(filepath, subset, "rgb")
        rgb_name_list = os.listdir(rgb_path)
        rgb_name_list.sort()

        for i in range(int(len(rgb_name_list)/2)):
            left_img.append(os.path.join(rgb_path, rgb_name_list[i*2]))
            right_img.append(os.path.join(rgb_path, rgb_name_list[i*2+1]))

    return left_img, right_img, masks

def listfile_stage2_train(input_path, gt_path, soil):

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
    # soil_mask = []
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

    ep_list = os.listdir(gt_path)      #ep子文件夹名列表
    ep_list.sort()
    
    for ep in ep_list:
        for subset in ['training', 'validation']:
            if soil:
                disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth_soiled")
                conf_map_path = os.path.join(input_path, ep, subset, "conf_map_soiled")
                rgb_path = os.path.join(gt_path, ep, subset, "rgb_soiled")
            else:
                disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth")
                conf_map_path = os.path.join(input_path, ep, subset, "conf_map")
                rgb_path = os.path.join(gt_path, ep, subset, "rgb")
            
            depth_path = os.path.join(gt_path, ep, subset, "depth")
            # if subset == 'training':
            #     soil_mask_path = os.path.join(gt_path, ep, subset, "soil_mask")
            #     soil_mask_name_list = os.listdir(soil_mask_path)
            #     soil_mask_name_list.sort()
            
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
                    # soil_mask.append(os.path.join(soil_mask_path, soil_mask_name_list[frame]))
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

def listfile_stage2_train_omnifisheye(input_path, gt_path):

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
    # soil_mask = []
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
    
    for subset in ['training', 'testing']:
        disp_pred2depth_path = os.path.join(input_path, subset, "disp_pred2depth")
        conf_map_path = os.path.join(input_path, subset, "conf_map")
        rgb_path = os.path.join(gt_path, subset, "rgb")
        depth_path = os.path.join(gt_path, subset, "depth")
        
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

def listfile_stage2_train_3d60(input_path):

    train_12=[]
    train_13=[]
    train_12_conf=[]
    train_13_conf=[]
    train_rgb1=[]
    train_rgb2=[]
    train_gt = []
    # soil_mask = []
    val_12=[]
    val_13=[]
    val_12_conf=[]
    val_13_conf=[]
    val_rgb1=[]
    val_rgb2=[]
    val_gt = []
    
    for subset in ['training', 'validation']:
        disp_pred2depth_path = os.path.join(input_path, subset, "disp_pred2depth")
        conf_map_path = os.path.join(input_path, subset, "conf_map")
        rgb_path = os.path.join(input_path, subset, "rgb")
        depth_path = os.path.join(input_path, subset, "depth")
        
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
                train_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*2]))
                train_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*2+1]))
                train_12_conf.append(os.path.join(conf_map_path, conf_name_list[frame*2]))
                train_13_conf.append(os.path.join(conf_map_path, conf_name_list[frame*2+1]))
                train_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame*2]))
                train_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame*2+1]))
                train_gt.append(os.path.join(depth_path, depth_name_list[frame]))
        else:
            for frame in range(len(depth_name_list)):
                val_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*2]))
                val_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*2+1]))
                val_12_conf.append(os.path.join(conf_map_path, conf_name_list[frame*2]))
                val_13_conf.append(os.path.join(conf_map_path, conf_name_list[frame*2+1]))
                val_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame*2]))
                val_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame*2+1]))
                val_gt.append(os.path.join(depth_path, depth_name_list[frame]))

    train_depthes = [train_12, train_13]
    train_confs = [train_12_conf, train_13_conf]
    train_rgbs = [train_rgb1, train_rgb2]
    val_depthes = [val_12, val_13]
    val_confs = [val_12_conf, val_13_conf]
    val_rgbs = [val_rgb1, val_rgb2]
    return train_depthes, train_confs, train_rgbs, train_gt, val_depthes, val_confs, val_rgbs, val_gt

def listfile_stage2_test(input_path, gt_path, soil):
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

    ep_list = os.listdir(gt_path)      #ep子文件夹名列表
    ep_list.sort()
    subset = "testing"

    if soil:
        for soil_type_dir in ['glare/', 'mud/', 'water/']:
            soil_type_path = "testing/disp_pred2depth_soiled/" + soil_type_dir
            soil_type_conf_path = "testing/conf_map_soiled/" + soil_type_dir
            soil_type_rgb_path = "testing/rgb_soiled/" + soil_type_dir
            soil_type_gt_path = "testing/depth_for_soiled/" + soil_type_dir

            for soil_cam_num_dir in ['1_soiled_cam/', '2_soiled_cam/']:
                soil_cam_num_path = soil_type_path + soil_cam_num_dir
                soil_cam_num_conf_path = soil_type_conf_path + soil_cam_num_dir
                soil_cam_num_rgb_path = soil_type_rgb_path + soil_cam_num_dir
                soil_cam_num_gt_path = soil_type_gt_path + soil_cam_num_dir

                for soil_num_dir in ['2_spot/', '3_spot/', '4_spot/', '5_spot/', '6_spot/']:
                    soil_num_path = soil_cam_num_path + soil_num_dir
                    soil_num_conf_path = soil_cam_num_conf_path + soil_num_dir
                    soil_num_rgb_path = soil_cam_num_rgb_path + soil_num_dir
                    soil_num_gt_path = soil_cam_num_gt_path + soil_num_dir
                    
                    for soil_rate_dir in ['05percent/', '10percent/', '15percent/', '20percent/']:
                        soil_rate_path = soil_num_path + soil_rate_dir
                        soil_rate_conf_path = soil_num_conf_path + soil_rate_dir
                        soil_rate_rgb_path = soil_num_rgb_path + soil_rate_dir
                        soil_rate_gt_path = soil_num_gt_path + soil_rate_dir
                        
                        for ep in ep_list:
                            disp_pred2depth_path = os.path.join(input_path, ep, soil_rate_path)
                            input_name_list = os.listdir(disp_pred2depth_path)
                            input_name_list.sort()

                            conf_path = os.path.join(input_path, ep, soil_rate_conf_path)
                            conf_name_list = os.listdir(conf_path)
                            conf_name_list.sort()

                            depth_path = os.path.join(gt_path, ep, soil_rate_gt_path)
                            depth_name_list = os.listdir(depth_path)
                            depth_name_list.sort()

                            rgb_path = os.path.join(gt_path, ep, soil_rate_rgb_path)
                            rgb_name_list = os.listdir(rgb_path)
                            rgb_name_list.sort()

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
    else:
        for ep in ep_list:
            disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth")
            input_name_list = os.listdir(disp_pred2depth_path)
            input_name_list.sort()
            conf_path = os.path.join(input_path, ep, subset, "conf_map")
            conf_name_list = os.listdir(conf_path)
            conf_name_list.sort()
            depth_path = os.path.join(gt_path, ep, subset, "depth")
            depth_name_list = os.listdir(depth_path)
            depth_name_list.sort()
            rgb_path = os.path.join(gt_path, ep, subset, "rgb")
            rgb_name_list = os.listdir(rgb_path)
            rgb_name_list.sort()
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

def listfile_stage2_test_omnifisheye(input_path, gt_path):
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

    subset = "testing"

    disp_pred2depth_path = os.path.join(input_path, subset, "disp_pred2depth")
    input_name_list = os.listdir(disp_pred2depth_path)
    input_name_list.sort()
    conf_path = os.path.join(input_path, subset, "conf_map")
    conf_name_list = os.listdir(conf_path)
    conf_name_list.sort()
    depth_path = os.path.join(gt_path, subset, "depth")
    depth_name_list = os.listdir(depth_path)
    depth_name_list.sort()
    rgb_path = os.path.join(gt_path, subset, "rgb")
    rgb_name_list = os.listdir(rgb_path)
    rgb_name_list.sort()
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

def listfile_stage2_test_3d60(input_path):
    test_12=[]
    test_13=[]
    test_12_conf=[]
    test_13_conf=[]
    test_rgb1=[]
    test_rgb2=[]
    test_gt = []

    subset = "testing"

    disp_pred2depth_path = os.path.join(input_path, subset, "disp_pred2depth")
    input_name_list = os.listdir(disp_pred2depth_path)
    input_name_list.sort()
    conf_path = os.path.join(input_path, subset, "conf_map")
    conf_name_list = os.listdir(conf_path)
    conf_name_list.sort()
    depth_path = os.path.join(input_path, subset, "depth")
    depth_name_list = os.listdir(depth_path)
    depth_name_list.sort()
    rgb_path = os.path.join(input_path, subset, "rgb")
    rgb_name_list = os.listdir(rgb_path)
    rgb_name_list.sort()
    for frame in range(len(depth_name_list)):
        test_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*2]))
        test_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*2+1]))
        test_12_conf.append(os.path.join(conf_path, conf_name_list[frame*2]))
        test_13_conf.append(os.path.join(conf_path, conf_name_list[frame*2+1]))
        test_gt.append(os.path.join(depth_path, depth_name_list[frame]))
        test_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame*2]))
        test_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame*2+1]))

    test_depthes = [test_12, test_13]
    test_confs = [test_12_conf, test_13_conf]
    test_rgbs = [test_rgb1, test_rgb2]
    return test_depthes, test_confs, test_rgbs, test_gt

def listfile_stage2_output(input_path, gt_path, soil):
    if soil:
        depth_path = input_path+'npy/soiled/'
        rgb_path = input_path+'rgb/soiled/'
    else:
        depth_path = input_path+'npy/clean/'
        rgb_path = input_path+'rgb/clean/'
    gt_path = gt_path+'npy/'

    test_12=[]
    test_13=[]
    test_14=[]
    test_23=[]
    test_24=[]
    test_34=[]
    test_rgb1=[]
    test_rgb2=[]
    test_rgb3=[]
    test_rgb4=[]
    test_depth = []
    
    input_name_list = os.listdir(depth_path)
    input_name_list.sort()
    rgb_name_list = os.listdir(rgb_path)
    rgb_name_list.sort()
    depth_name_list = os.listdir(gt_path)
    depth_name_list.sort()

    for frame in range(len(depth_name_list)):
        test_12.append(os.path.join(depth_path, input_name_list[frame*6]))
        test_13.append(os.path.join(depth_path, input_name_list[frame*6+1]))
        test_14.append(os.path.join(depth_path, input_name_list[frame*6+2]))
        test_23.append(os.path.join(depth_path, input_name_list[frame*6+3]))
        test_24.append(os.path.join(depth_path, input_name_list[frame*6+4]))
        test_34.append(os.path.join(depth_path, input_name_list[frame*6+5]))
        test_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame*4]))
        test_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame*4+1]))
        test_rgb3.append(os.path.join(rgb_path, rgb_name_list[frame*4+2]))
        test_rgb4.append(os.path.join(rgb_path, rgb_name_list[frame*4+3]))
        test_depth.append(os.path.join(gt_path, depth_name_list[frame]))

    return test_12, test_13, test_14, test_23, test_24, test_34, test_rgb1, test_rgb2, test_rgb3, test_rgb4, test_depth