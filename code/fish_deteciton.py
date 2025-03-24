import pyrealsense2 as rs
import numpy as np
import cv2
import re
from ultralytics import YOLO
import csv

def main():
    try:
        # RealSenseのパイプラインをセットアップ
        pipeline = rs.pipeline()
        config = rs.config()
        
        bag_file_path = '20240808_144616.bag'  # ここにbagファイルのパスを指定
        config.enable_device_from_file(bag_file_path)
        
        pipeline.start(config)

        # 再生速度を制御するためのplaybackオブジェクトを取得
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)  # リアルタイム再生を無効にする
        
        model = YOLO("model_fish.pt")

        frame_count = 0
        detection_data = []  # 検出データを収集するリスト

        # ビデオライターをセットアップ
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックを指定
        out = cv2.VideoWriter('output.mov', fourcc, 30.0, (640, 480))  # 出力ファイル名、コーデック、フレームレート、フレームサイズ

        while True:
            # フレームのセットを取得
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
            if not depth_frame or not color_frame:
                continue

            # 深度フレームとカラー（RGB）フレームをnumpy配列に変換
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # RGB画像をBGRに変換
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # YOLOで物体検出
            results = model.track(color_image, persist=True, show=False)
            
            for result in results:
                if result.masks is not None and hasattr(result.masks, 'data'):  # result.masksがNoneでなく、data属性を持つことを確認
                    for idx, mask in enumerate(result.masks.data):
                        # マスクからピクセル数をカウント
                        mask = mask.cpu().numpy()  # マスクをnumpy配列に変換
                        pixel_count = np.count_nonzero(mask)

                        distances = []

                        for point in np.argwhere(mask):  # マスクの中のピクセル位置を取得
                            y, x = point  # argwhereは(y, x)の順で返される
                            if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:  # 座標の範囲チェック
                                distance = depth_frame.get_distance(x, y)
                                if not np.isnan(distance):  # NaNをチェック
                                    distances.append(distance)

                        if distances:
                            avg_distance = np.mean(distances)
                        else:
                            avg_distance = 0

                        # 検出結果を表示
                        center_x = np.mean(np.argwhere(mask)[:, 1])
                        center_y = np.mean(np.argwhere(mask)[:, 0])

                        # NaNチェックを追加して整数に変換
                        if np.isnan(center_x) or np.isnan(center_y):
                            center_x, center_y = 0, 0
                        else:
                            center_x, center_y = int(center_x), int(center_y)

                        label = f"{avg_distance:.2f}m"

                        # 検出結果を保存
                        if hasattr(result.boxes, 'id'):  # result.boxesがid属性を持つことを確認
                            detection_id = result.boxes.id[idx] if idx < len(result.boxes.id) else -1
                        else:
                            detection_id = -1
                        
                        detection_id= str(detection_id)
                        pattern = r'\d+'
                        detection_id_str= re.findall(pattern, detection_id)
                        input_string = str(detection_id_str)
                        detection_id_num = int(input_string.strip("[]'"))
                        detection_data.append([detection_id_num, frame_count, avg_distance, pixel_count])
                        
                        # テキストを描画
                        cv2.putText(color_image, label, (center_x + 5, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("YOLOv8トラッキング", results[0].plot())

                # フレームを保存
                frame_filename = f"frame_{frame_count:04d}.png"
                cv2.imwrite(frame_filename, results[0].plot())
                print(f"Saved frame: {frame_filename}")

            # ビデオファイルにフレームを書き込む
            out.write(results[0].plot())

            frame_count += 1

            # 'q'キーを押したら終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # クリーンアップ
        pipeline.stop()
        cv2.destroyAllWindows()
        out.release()

        # 検出データをCSVファイルに書き出し
        csv_filename = 'detection_data.csv'
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Detection ID", "Frame", "Average Distance (m)", "Pixel Count"])
            writer.writerows(detection_data)
        print(f"Saved detection data to {csv_filename}")

if __name__ == "__main__":
    main()
