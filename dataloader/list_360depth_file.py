import os

def listfile(filepath):

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    ep_list = [ep for ep in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, ep))]   #ep子文件夹名列表

    for ep in ep_list:
        rgb_path = os.path.join(filepath, ep, "rgb")
        disp_path = os.path.join(filepath, ep, "disp")
        
        rgb_name_list = os.listdir(rgb_path)
        rgb_name_list.sort()
        disp_name_list = os.listdir(disp_path)
        disp_name_list.sort()
      
        for frame in range(100):
            if frame % 10 != 9:
                for cam_pair in range(6):
                    all_left_img.append(os.path.join(rgb_path, rgb_name_list[frame*12+cam_pair*2]))
                    all_right_img.append(os.path.join(rgb_path, rgb_name_list[frame*12+cam_pair*2+1]))
                    all_left_disp.append(os.path.join(disp_path, disp_name_list[frame*6+cam_pair]))
            else:
                for cam_pair in range(6):
                    test_left_img.append(os.path.join(rgb_path, rgb_name_list[frame*12+cam_pair*2]))
                    test_right_img.append(os.path.join(rgb_path, rgb_name_list[frame*12+cam_pair*2+1]))
                    test_left_disp.append(os.path.join(disp_path, disp_name_list[frame*6+cam_pair]))


    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


def listfile_stage2(filepath, do_disp2depth):

    train_12=[]
    train_13=[]
    train_14=[]
    train_23=[]
    train_24=[]
    train_34=[]
    train_depth = []
    test_12=[]
    test_13=[]
    test_14=[]
    test_23=[]
    test_24=[]
    test_34=[]
    test_depth = []

    ep_list = [ep for ep in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, ep))]   #ep子文件夹名列表

    for ep in ep_list:
        if do_disp2depth:
            input_path = os.path.join(filepath, ep, "disp_pred")
        else:
            input_path = os.path.join(filepath, ep, "disp_pred2depth")
        depth_path = os.path.join(filepath, ep, "depth")
        
        input_name_list = os.listdir(input_path)
        input_name_list.sort()
        depth_name_list = os.listdir(depth_path)
        depth_name_list.sort()
      
        for frame in range(100):
            if frame % 10 != 9:
                train_12.append(os.path.join(input_path, input_name_list[frame*6]))
                train_13.append(os.path.join(input_path, input_name_list[frame*6+1]))
                train_14.append(os.path.join(input_path, input_name_list[frame*6+2]))
                train_23.append(os.path.join(input_path, input_name_list[frame*6+3]))
                train_24.append(os.path.join(input_path, input_name_list[frame*6+4]))
                train_34.append(os.path.join(input_path, input_name_list[frame*6+5]))
                train_depth.append(os.path.join(depth_path, depth_name_list[frame]))
            else:
                test_12.append(os.path.join(input_path, input_name_list[frame*6]))
                test_13.append(os.path.join(input_path, input_name_list[frame*6+1]))
                test_14.append(os.path.join(input_path, input_name_list[frame*6+2]))
                test_23.append(os.path.join(input_path, input_name_list[frame*6+3]))
                test_24.append(os.path.join(input_path, input_name_list[frame*6+4]))
                test_34.append(os.path.join(input_path, input_name_list[frame*6+5]))
                test_depth.append(os.path.join(depth_path, depth_name_list[frame]))

    return train_12, train_13, train_14, train_23, train_24, train_34, train_depth, test_12, test_13, test_14, test_23, test_24, test_34, test_depth