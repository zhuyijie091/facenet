#----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time

import cv2
import numpy as np

from retinaface import Retinaface

if __name__ == "__main__":
    retinaface = Retinaface()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        '''
        predict.py有几个注意点
        1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
        2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
        3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
        4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
        5、在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
        '''
        while True:
            img = input('Input image filename:')
            image = cv2.imread(img)
            if image is None:
                print('Open Error! Try again!')
                continue
            else:
                image   = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)
                r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                # cv2.imshow("after",r_image)
                cv2.imwrite("img.jpg", r_image)
                cv2.waitKey(0)

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 进行检测
            frame = np.array(retinaface.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = cv2.imread('img/obama.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        tact_time = retinaface.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    # elif mode == "dir_predict":
    #     import os
    #
    #     from tqdm import tqdm
    #
    #     img_names = os.listdir(dir_origin_path)
    #     for img_name in tqdm(img_names):
    #         if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #             image_path  = os.path.join(dir_origin_path, img_name)
    #             image       = cv2.imread(image_path)
    #             image       = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #             r_image     = retinaface.detect_image(image)
    #             r_image     = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)
    # elif mode == "dir_predict":
    #     import os
    #     from tqdm import tqdm
    #
    #     # 初始化评估数据存储
    #     true_labels = []
    #     predicted_labels = []
    #     error_log = []
    #
    #     # 遍历所有子文件夹
    #     for root, dirs, files in os.walk(dir_origin_path):
    #         for file in tqdm(files, desc=f"processing {root}"):
    #             if file.lower().endswith(
    #                     ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #                 try:
    #                     # 图像路径处理
    #                     image_path = os.path.join(root, file)
    #
    #                     # 获取真实标签（使用父文件夹名称）
    #                     true_label = os.path.basename(root)
    #
    #                     # 检查文件是否存在
    #                     if not os.path.exists(image_path):
    #                         raise FileNotFoundError(f"文件不存在: {image_path}")
    #
    #                     # 读取并处理图像
    #                     image = cv2.imread(image_path)
    #                     if image is None:
    #                         raise ValueError(f"无法读取图像: {image_path}")
    #
    #                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #                     # 进行预测
    #                     r_image, face_names = retinaface.detect_image(image)
    #
    #                     # 预测标签处理（取出现次数最多的标签）
    #                     pred_label = "unknown"
    #                     if len(face_names) > 0:
    #                         pred_label = max(set(face_names), key=face_names.count)
    #
    #                     # 存储评估数据
    #                     true_labels.append(true_label)
    #                     predicted_labels.append(pred_label)
    #
    #                     # 打印当前文件的预测结果
    #                     print(f"文件: {image_path} | 真实标签: {true_label} | 预测标签: {pred_label}")
    #
    #                 except Exception as e:
    #                     error_msg = f"{image_path} | 错误信息: {str(e)}"
    #                     error_log.append(error_msg)
    #                     print(error_msg)
    #
    #     # 输出评估结果
    #     if true_labels and predicted_labels:
    #         from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    #
    #         # 计算指标
    #         accuracy = accuracy_score(true_labels, predicted_labels)
    #         report = classification_report(true_labels, predicted_labels)
    #         conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #
    #         # 打印分类报告和混淆矩阵
    #         print(f"\n准确率: {accuracy:.4f}")
    #         print("\n分类报告:")
    #         print(report)
    #         print("\n混淆矩阵:")
    #         print(conf_matrix)
    #
    #     # 打印错误日志（如果有）
    #     if error_log:
    #         print("\n错误汇总:")
    #         for error in error_log:
    #             print(error)
    elif mode == "dir_predict":
        import os
        import time
        from tqdm import tqdm

        # 初始化评估数据存储
        true_labels = []
        predicted_labels = []
        error_log = []
        total_files = 0  # 新增：文件计数器

        # 记录开始时间
        start_total = time.time()

        # 遍历所有子文件夹
        for root, dirs, files in os.walk(dir_origin_path):
            for file in tqdm(files, desc=f"Processing {root}"):
                if file.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    total_files += 1  # 新增：计数有效文件
                    image_path = os.path.join(root, file)
                    try:
                        # 获取真实标签（使用父文件夹名称）
                        true_label = os.path.basename(root)

                        # 检查文件是否存在
                        if not os.path.exists(image_path):
                            raise FileNotFoundError(f"文件不存在: {image_path}")

                        # 读取并处理图像
                        image = cv2.imread(image_path)
                        if image is None:
                            raise ValueError(f"无法读取图像: {image_path}")

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # 进行预测（记录单个文件处理时间）
                        file_start = time.time()  # 可选：单个文件计时
                        r_image, face_names = retinaface.detect_image(image)
                        file_time = time.time() - file_start  # 可选：单个文件计时

                        # 预测标签处理（取出现次数最多的标签）
                        pred_label = "unknown"
                        if len(face_names) > 0:
                            pred_label = max(set(face_names), key=face_names.count)

                        # 存储评估数据
                        true_labels.append(true_label)
                        predicted_labels.append(pred_label)

                        # 打印结果（添加处理时间）
                        print(f"文件: {image_path} | 真实: {true_label} | 预测: {pred_label} | 耗时: {file_time:.2f}s")

                    except Exception as e:
                        error_msg = f"{image_path} | 错误信息: {str(e)}"
                        error_log.append(error_msg)
                        print(error_msg)

        # 计算总耗时
        total_time = time.time() - start_total

        # 输出评估结果
        if true_labels and predicted_labels:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

            accuracy = accuracy_score(true_labels, predicted_labels)
            report = classification_report(true_labels, predicted_labels)
            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            print(f"\n准确率: {accuracy:.4f}")
            print("\n分类报告:")
            print(report)
            print("\n混淆矩阵:")
            print(conf_matrix)

        # 输出性能统计
        print("\n性能统计")
        print("▔" * 30)
        print(f"总检测时长: {total_time:.2f} 秒")
        print(f"处理文件总数: {total_files} 个")
        print(f"成功处理: {len(true_labels)} 个")
        print(f"失败文件: {len(error_log)} 个")
        if total_files > 0:
            print(f"平均耗时: {total_time / total_files:.4f} 秒/个")

        # 输出错误日志
        if error_log:
            print("\n错误汇总:")
            for error in error_log:
                print(error)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
