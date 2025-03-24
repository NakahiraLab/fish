import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv('detection_data.csv')

# 'Average Distance (m)' が0でない行のみを残す
df = df[df['Average Distance (m)'] != 0]

# 固定値
FOV_horizontal = 86  # 水平方向の視野角（度）
FOV_vertical = 57    # 垂直方向の視野角（度）
resolution_horizontal = 840  # 水平方向の解像度（ピクセル）
resolution_vertical = 480    # 垂直方向の解像度（ピクセル）

# 視野幅の計算
# 水平方向の視野幅を計算
df['W_horizontal'] = 2 * df['Average Distance (m)'] * math.tan(math.radians(FOV_horizontal / 2))
# 垂直方向の視野幅を計算
df['W_vertical'] = 2 * df['Average Distance (m)'] * math.tan(math.radians(FOV_vertical / 2))

# 1ピクセルあたりの物理的な幅の計算
# chatgptとネットの情報を参考に以下の計算式構築したが、少し怪しいため要検証
df['pixel_width'] = df['W_horizontal'] / resolution_horizontal  # 水平方向
df['pixel_height'] = df['W_vertical'] / resolution_vertical  # 垂直方向

# 物体の物理的な大きさ（面積）を計算
df['size'] = df['Pixel Count'] * df['pixel_width'] * df['pixel_height']

# 'size' が0でない行でグループ化して平均を計算
mean_pixel_counts = df[df['size'] != 0].groupby('Detection ID')['size'].mean()
print(mean_pixel_counts)

# ヒストグラムを計算
bins = np.linspace(0, 2, 5)
freq, _ = np.histogram(mean_pixel_counts, bins=bins)
class_value = (bins[:-1] + bins[1:]) / 2  # 階級値
rel_freq = freq / mean_pixel_counts.count()  # 相対度数
cum_freq = np.cumsum(freq)  # 累積度数

# データフレームに結果をまとめる
dist = pd.DataFrame(
    {
        "grade value": class_value,
        "meter": freq,
        "相対度数": rel_freq,
        "累積度数": cum_freq,
    }
)

print(dist)

# 棒グラフをプロット
dist.plot.bar(x="grade value", y="meter", width=1, ec="k", lw=2)
plt.show()
