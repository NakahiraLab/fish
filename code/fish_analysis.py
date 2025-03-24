import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV�t�@�C����ǂݍ���
df = pd.read_csv('detection_data.csv')

# 'Average Distance (m)' ��0�łȂ��s�݂̂��c��
df = df[df['Average Distance (m)'] != 0]

# �Œ�l
FOV_horizontal = 86  # ���������̎���p�i�x�j
FOV_vertical = 57    # ���������̎���p�i�x�j
resolution_horizontal = 840  # ���������̉𑜓x�i�s�N�Z���j
resolution_vertical = 480    # ���������̉𑜓x�i�s�N�Z���j

# ���앝�̌v�Z
# ���������̎��앝���v�Z
df['W_horizontal'] = 2 * df['Average Distance (m)'] * math.tan(math.radians(FOV_horizontal / 2))
# ���������̎��앝���v�Z
df['W_vertical'] = 2 * df['Average Distance (m)'] * math.tan(math.radians(FOV_vertical / 2))

# 1�s�N�Z��������̕����I�ȕ��̌v�Z
df['pixel_width'] = df['W_horizontal'] / resolution_horizontal  # ��������
df['pixel_height'] = df['W_vertical'] / resolution_vertical  # ��������

# ���̂̕����I�ȑ傫���i�ʐρj���v�Z
df['size'] = df['Pixel Count'] * df['pixel_width'] * df['pixel_height']

# 'size' ��0�łȂ��s�ŃO���[�v�����ĕ��ς��v�Z
mean_pixel_counts = df[df['size'] != 0].groupby('Detection ID')['size'].mean()
print(mean_pixel_counts)

# �q�X�g�O�������v�Z
bins = np.linspace(0, 2, 5)
freq, _ = np.histogram(mean_pixel_counts, bins=bins)
class_value = (bins[:-1] + bins[1:]) / 2  # �K���l
rel_freq = freq / mean_pixel_counts.count()  # ���Γx��
cum_freq = np.cumsum(freq)  # �ݐϓx��

# �f�[�^�t���[���Ɍ��ʂ��܂Ƃ߂�
dist = pd.DataFrame(
    {
        "grade value": class_value,
        "meter": freq,
        "���Γx��": rel_freq,
        "�ݐϓx��": cum_freq,
    }
)

print(dist)

# �_�O���t���v���b�g
dist.plot.bar(x="grade value", y="meter", width=1, ec="k", lw=2)
plt.show()
