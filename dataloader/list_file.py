import os

"""
NOTE:

Multi-view 360-degree Dataset;
3000 frames;
each frame contains 6 pairs of left-right cassini preojection images
(12,13,14,23,24,34), 6 disparity maps and 1 depth maps referenced to the coordinate system of camera 1;

Dataset directory structure:
Deep360
+-- README.txt
+-- ep1_500frames
|   +-- training (350 frames)
|   |   +-- rgb (each frame consists of 6 pairs of rectified panoramas)
|   |   +-- rgb_soiled (soiled panoramas)
|   |   +-- disp (each frame consists of 6 disparity maps)
|   |   +-- depth (each frame consists of 1 ground truth depth map)
|   +-- validation (50 frames)
|   +-- testing (100 frames)
+-- ep2_500frames
+-- ep3_500frames
+-- ep4_500frames
+-- ep5_500frames
+-- ep6_500frames

"""

def list_deep360_disparity_train(filepath, soiled):

  train_left_img = []
  train_right_img = []
  train_left_disp = []
  val_left_img = []
  val_right_img = []
  val_left_disp = []

  ep_list = ["ep%d_500frames" % i for i in range(1, 7)]
  ep_list.sort()

  rgb_dir = "rgb_soiled" if soiled else "rgb"

  for ep in ep_list:
    for subset in ['training', 'validation']:
      rgb_path = os.path.join(filepath, ep, subset, rgb_dir)
      disp_path = os.path.join(filepath, ep, subset, "disp")

      rgb_name_list = os.listdir(rgb_path)
      rgb_name_list.sort()
      disp_name_list = os.listdir(disp_path)
      disp_name_list.sort()

      if subset == 'training':
        for i in range(len(disp_name_list)):
          train_left_img.append(os.path.join(rgb_path, rgb_name_list[i * 2]))
          train_right_img.append(os.path.join(rgb_path, rgb_name_list[i * 2 + 1]))
          train_left_disp.append(os.path.join(disp_path, disp_name_list[i]))
      else:
        for i in range(len(disp_name_list)):
          val_left_img.append(os.path.join(rgb_path, rgb_name_list[i * 2]))
          val_right_img.append(os.path.join(rgb_path, rgb_name_list[i * 2 + 1]))
          val_left_disp.append(os.path.join(disp_path, disp_name_list[i]))

  return train_left_img, train_right_img, train_left_disp, val_left_img, val_right_img, val_left_disp


def list_deep360_disparity_test(filepath, soiled):

  test_left_img = []
  test_right_img = []
  test_left_disp = []

  ep_list = ["ep%d_500frames" % i for i in range(1, 7)]
  ep_list.sort()

  rgb_dir = "rgb_soiled" if soiled else "rgb"

  subset = "testing"
  for ep in ep_list:
    rgb_path = os.path.join(filepath, ep, subset, rgb_dir)
    disp_path = os.path.join(filepath, ep, subset, "disp")

    rgb_name_list = os.listdir(rgb_path)
    rgb_name_list.sort()
    disp_name_list = os.listdir(disp_path)
    disp_name_list.sort()

    for i in range(len(disp_name_list)):
      test_left_img.append(os.path.join(rgb_path, rgb_name_list[i * 2]))
      test_right_img.append(os.path.join(rgb_path, rgb_name_list[i * 2 + 1]))
      test_left_disp.append(os.path.join(disp_path, disp_name_list[i]))

  return test_left_img, test_right_img, test_left_disp


def list_deep360_fusion_train(input_path, dataset_path, soil):
  train_12 = []
  train_13 = []
  train_14 = []
  train_23 = []
  train_24 = []
  train_34 = []
  train_12_conf = []
  train_13_conf = []
  train_14_conf = []
  train_23_conf = []
  train_24_conf = []
  train_34_conf = []
  train_rgb1 = []
  train_rgb2 = []
  train_rgb3 = []
  train_rgb4 = []
  train_gt = []

  val_12 = []
  val_13 = []
  val_14 = []
  val_23 = []
  val_24 = []
  val_34 = []
  val_12_conf = []
  val_13_conf = []
  val_14_conf = []
  val_23_conf = []
  val_24_conf = []
  val_34_conf = []
  val_rgb1 = []
  val_rgb2 = []
  val_rgb3 = []
  val_rgb4 = []
  val_gt = []

  ep_list = ["ep%d_500frames" % i for i in range(1, 7)]
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
          train_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6]))
          train_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 1]))
          train_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 2]))
          train_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 3]))
          train_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 4]))
          train_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 5]))
          train_12_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6]))
          train_13_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 1]))
          train_14_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 2]))
          train_23_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 3]))
          train_24_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 4]))
          train_34_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 5]))
          train_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame * 12]))
          train_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 1]))
          train_rgb3.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 10]))
          train_rgb4.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 11]))
          train_gt.append(os.path.join(depth_path, depth_name_list[frame]))
      else:
        for frame in range(len(depth_name_list)):
          val_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6]))
          val_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 1]))
          val_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 2]))
          val_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 3]))
          val_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 4]))
          val_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 5]))
          val_12_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6]))
          val_13_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 1]))
          val_14_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 2]))
          val_23_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 3]))
          val_24_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 4]))
          val_34_conf.append(os.path.join(conf_map_path, conf_name_list[frame * 6 + 5]))
          val_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame * 12]))
          val_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 1]))
          val_rgb3.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 10]))
          val_rgb4.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 11]))
          val_gt.append(os.path.join(depth_path, depth_name_list[frame]))

  train_depthes = [train_12, train_13, train_14, train_23, train_24, train_34]
  train_confs = [train_12_conf, train_13_conf, train_14_conf, train_23_conf, train_24_conf, train_34_conf]
  train_rgbs = [train_rgb1, train_rgb2, train_rgb3, train_rgb4]
  val_depthes = [val_12, val_13, val_14, val_23, val_24, val_34]
  val_confs = [val_12_conf, val_13_conf, val_14_conf, val_23_conf, val_24_conf, val_34_conf]
  val_rgbs = [val_rgb1, val_rgb2, val_rgb3, val_rgb4]
  return train_depthes, train_confs, train_rgbs, train_gt, val_depthes, val_confs, val_rgbs, val_gt


def list_deep360_fusion_test(input_path, dataset_path, soil):
  test_12 = []
  test_13 = []
  test_14 = []
  test_23 = []
  test_24 = []
  test_34 = []
  test_12_conf = []
  test_13_conf = []
  test_14_conf = []
  test_23_conf = []
  test_24_conf = []
  test_34_conf = []
  test_rgb1 = []
  test_rgb2 = []
  test_rgb3 = []
  test_rgb4 = []
  test_gt = []

  ep_list = ["ep%d_500frames" % i for i in range(1, 7)]
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
      test_12.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6]))
      test_13.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 1]))
      test_14.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 2]))
      test_23.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 3]))
      test_24.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 4]))
      test_34.append(os.path.join(disp_pred2depth_path, input_name_list[frame * 6 + 5]))
      test_12_conf.append(os.path.join(conf_path, conf_name_list[frame * 6]))
      test_13_conf.append(os.path.join(conf_path, conf_name_list[frame * 6 + 1]))
      test_14_conf.append(os.path.join(conf_path, conf_name_list[frame * 6 + 2]))
      test_23_conf.append(os.path.join(conf_path, conf_name_list[frame * 6 + 3]))
      test_24_conf.append(os.path.join(conf_path, conf_name_list[frame * 6 + 4]))
      test_34_conf.append(os.path.join(conf_path, conf_name_list[frame * 6 + 5]))
      test_gt.append(os.path.join(depth_path, depth_name_list[frame]))
      test_rgb1.append(os.path.join(rgb_path, rgb_name_list[frame * 12]))
      test_rgb2.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 1]))
      test_rgb3.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 10]))
      test_rgb4.append(os.path.join(rgb_path, rgb_name_list[frame * 12 + 11]))

  test_depthes = [test_12, test_13, test_14, test_23, test_24, test_34]
  test_confs = [test_12_conf, test_13_conf, test_14_conf, test_23_conf, test_24_conf, test_34_conf]
  test_rgbs = [test_rgb1, test_rgb2, test_rgb3, test_rgb4]
  return test_depthes, test_confs, test_rgbs, test_gt
