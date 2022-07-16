import os


def listfile_disparity_train(filepath, soiled):

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


def listfile_disparity_test(filepath, soiled):

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