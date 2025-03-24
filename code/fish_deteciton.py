import pyrealsense2 as rs
import numpy as np
import cv2
import re
from ultralytics import YOLO
import csv

def main():
    try:
        # RealSense�̃p�C�v���C�����Z�b�g�A�b�v
        pipeline = rs.pipeline()
        config = rs.config()
        
        bag_file_path = '20240808_144616.bag'  # ������bag�t�@�C���̃p�X���w��
        config.enable_device_from_file(bag_file_path)
        
        pipeline.start(config)

        # �Đ����x�𐧌䂷�邽�߂�playback�I�u�W�F�N�g���擾
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)  # ���A���^�C���Đ��𖳌��ɂ���
        
        model = YOLO("model_fish.pt")

        frame_count = 0
        detection_data = []  # ���o�f�[�^�����W���郊�X�g

        # �r�f�I���C�^�[���Z�b�g�A�b�v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # �R�[�f�b�N���w��
        out = cv2.VideoWriter('output.mov', fourcc, 30.0, (640, 480))  # �o�̓t�@�C�����A�R�[�f�b�N�A�t���[�����[�g�A�t���[���T�C�Y

        while True:
            # �t���[���̃Z�b�g���擾
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
            if not depth_frame or not color_frame:
                continue

            # �[�x�t���[���ƃJ���[�iRGB�j�t���[����numpy�z��ɕϊ�
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # RGB�摜��BGR�ɕϊ�
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # YOLO�ŕ��̌��o
            results = model.track(color_image, persist=True, show=False)
            
            for result in results:
                if result.masks is not None and hasattr(result.masks, 'data'):  # result.masks��None�łȂ��Adata�����������Ƃ��m�F
                    for idx, mask in enumerate(result.masks.data):
                        # �}�X�N����s�N�Z�������J�E���g
                        mask = mask.cpu().numpy()  # �}�X�N��numpy�z��ɕϊ�
                        pixel_count = np.count_nonzero(mask)

                        distances = []

                        for point in np.argwhere(mask):  # �}�X�N�̒��̃s�N�Z���ʒu���擾
                            y, x = point  # argwhere��(y, x)�̏��ŕԂ����
                            if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:  # ���W�͈̔̓`�F�b�N
                                distance = depth_frame.get_distance(x, y)
                                if not np.isnan(distance):  # NaN���`�F�b�N
                                    distances.append(distance)

                        if distances:
                            avg_distance = np.mean(distances)
                        else:
                            avg_distance = 0

                        # ���o���ʂ�\��
                        center_x = np.mean(np.argwhere(mask)[:, 1])
                        center_y = np.mean(np.argwhere(mask)[:, 0])

                        # NaN�`�F�b�N��ǉ����Đ����ɕϊ�
                        if np.isnan(center_x) or np.isnan(center_y):
                            center_x, center_y = 0, 0
                        else:
                            center_x, center_y = int(center_x), int(center_y)

                        label = f"{avg_distance:.2f}m"

                        # ���o���ʂ�ۑ�
                        if hasattr(result.boxes, 'id'):  # result.boxes��id�����������Ƃ��m�F
                            detection_id = result.boxes.id[idx] if idx < len(result.boxes.id) else -1
                        else:
                            detection_id = -1
                        
                        detection_id= str(detection_id)
                        pattern = r'\d+'
                        detection_id_str= re.findall(pattern, detection_id)
                        input_string = str(detection_id_str)
                        detection_id_num = int(input_string.strip("[]'"))
                        detection_data.append([detection_id_num, frame_count, avg_distance, pixel_count])
                        
                        # �e�L�X�g��`��
                        cv2.putText(color_image, label, (center_x + 5, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("YOLOv8�g���b�L���O", results[0].plot())

                # �t���[����ۑ�
                frame_filename = f"frame_{frame_count:04d}.png"
                cv2.imwrite(frame_filename, results[0].plot())
                print(f"Saved frame: {frame_filename}")

            # �r�f�I�t�@�C���Ƀt���[������������
            out.write(results[0].plot())

            frame_count += 1

            # 'q'�L�[����������I��
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # �N���[���A�b�v
        pipeline.stop()
        cv2.destroyAllWindows()
        out.release()

        # ���o�f�[�^��CSV�t�@�C���ɏ����o��
        csv_filename = 'detection_data.csv'
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Detection ID", "Frame", "Average Distance (m)", "Pixel Count"])
            writer.writerows(detection_data)
        print(f"Saved detection data to {csv_filename}")

if __name__ == "__main__":
    main()
