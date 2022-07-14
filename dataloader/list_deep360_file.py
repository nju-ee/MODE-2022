import os

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

def listfile_stage2_train(input_path, gt_path):

    train_12=[]
    train_13=[]
    train_14=[]
    train_23=[]
    train_24=[]
    train_34=[]
    train_depth = []
    val_12=[]
    val_13=[]
    val_14=[]
    val_23=[]
    val_24=[]
    val_34=[]
    val_depth = []

    ep_list = os.listdir(gt_path)      #ep子文件夹名列表
    ep_list.sort()
    
    for ep in ep_list:
        for subset in ['training', 'validation']:
            disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth")
            depth_path = os.path.join(gt_path, ep, subset, "depth")
            
            input_name_list = os.listdir(disp_pred2depth_path)
            input_name_list.sort()
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
                    train_depth.append(os.path.join(depth_path, depth_name_list[frame]))
            else:
                for frame in range(len(depth_name_list)):
                    val_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6]))
                    val_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+1]))
                    val_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+2]))
                    val_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+3]))
                    val_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+4]))
                    val_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+5]))
                    val_depth.append(os.path.join(depth_path, depth_name_list[frame]))

    return train_12, train_13, train_14, train_23, train_24, train_34, train_depth, val_12, val_13, val_14, val_23, val_24, val_34, val_depth

def listfile_stage2_test(input_path, gt_path):

    test_12=[]
    test_13=[]
    test_14=[]
    test_23=[]
    test_24=[]
    test_34=[]
    test_depth = []

    ep_list = os.listdir(gt_path)      #ep子文件夹名列表
    ep_list.sort()
    
    subset = "testing"
    for ep in ep_list:
        disp_pred2depth_path = os.path.join(input_path, ep, subset, "disp_pred2depth")
        depth_path = os.path.join(gt_path, ep, subset, "depth")
        
        input_name_list = os.listdir(disp_pred2depth_path)
        input_name_list.sort()
        depth_name_list = os.listdir(depth_path)
        depth_name_list.sort()

        for frame in range(len(depth_name_list)):
            test_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6]))
            test_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+1]))
            test_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+2]))
            test_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+3]))
            test_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+4]))
            test_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame*6+5]))
            test_depth.append(os.path.join(depth_path, depth_name_list[frame]))

    return test_12, test_13, test_14, test_23, test_24, test_34, test_depth

def listfile_stage2_output(input_path, gt_path):
    input_path = input_path+'npy/'
    gt_path = gt_path+'npy/'

    test_12=[]
    test_13=[]
    test_14=[]
    test_23=[]
    test_24=[]
    test_34=[]
    test_depth = []
    
    input_name_list = os.listdir(input_path)
    input_name_list.sort()
    depth_name_list = os.listdir(gt_path)
    depth_name_list.sort()

    for frame in range(len(depth_name_list)):
        test_12.append(os.path.join(input_path, input_name_list[frame*6]))
        test_13.append(os.path.join(input_path, input_name_list[frame*6+1]))
        test_14.append(os.path.join(input_path, input_name_list[frame*6+2]))
        test_23.append(os.path.join(input_path, input_name_list[frame*6+3]))
        test_24.append(os.path.join(input_path, input_name_list[frame*6+4]))
        test_34.append(os.path.join(input_path, input_name_list[frame*6+5]))
        test_depth.append(os.path.join(gt_path, depth_name_list[frame]))

    return test_12, test_13, test_14, test_23, test_24, test_34, test_depth