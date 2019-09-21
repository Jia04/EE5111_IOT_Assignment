import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def dynomodb_data():
    df = pd.read_csv('C:\\Users\\yaoji\\Dropbox\\EE5111\\ProjectDemo\\Data\\Patient2.csv')
    return df

df = dynomodb_data()

idx_lst = []
for i in range(0, df.shape[0]): 
    id = df['ID (S)'][i][11:]
    idx_lst.append(int(id))

df = pd.concat([pd.Series(idx_lst, name='Idx_ID'), df], axis=1)
# print(df.head())
data = df.sort_values('Idx_ID')
# print(data.head())

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.88, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='avr')
ax2 = fig.add_axes([0.1, 0.82, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='avf')
ax3 = fig.add_axes([0.1, 0.76, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='avl')
ax4 = fig.add_axes([0.1, 0.72, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='i')
ax5 = fig.add_axes([0.1, 0.66, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='ii')
ax6 = fig.add_axes([0.1, 0.60, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='iii')
ax7 = fig.add_axes([0.1, 0.54, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='v1')
ax8 = fig.add_axes([0.1, 0.48, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='v2')
ax9 = fig.add_axes([0.1, 0.42, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='v3')
ax10 = fig.add_axes([0.1, 0.36, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='v4')
ax11 = fig.add_axes([0.1, 0.30, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='v5')
ax12 = fig.add_axes([0.1, 0.24, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='v6')
ax13 = fig.add_axes([0.1, 0.18, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='vx')
ax14 = fig.add_axes([0.1, 0.12, 0.8, 0.06], xticklabels=[], ylim=(-1.5, 1.5), ylabel='vy')
ax15 = fig.add_axes([0.1, 0.06, 0.8, 0.06], ylim=(-1.5, 1.5), ylabel='vz')

x = np.linspace(0, 10)
ax1.plot(data['Lead_avr (N)'].values)
ax2.plot(data['Lead_avf (N)'].values)
ax3.plot(data['Lead_avl (N)'].values)
ax4.plot(data['Lead_i (N)'].values)
ax5.plot(data['Lead_ii (N)'].values)
ax6.plot(data['Lead_iii (N)'].values)
ax7.plot(data['Lead_v1 (N)'].values)
ax8.plot(data['Lead_v2 (N)'].values)
ax9.plot(data['Lead_v3 (N)'].values)
ax10.plot(data['Lead_v4 (N)'].values)
ax11.plot(data['Lead_v5 (N)'].values)
ax12.plot(data['Lead_v6 (N)'].values)
ax13.plot(data['Lead_vx (N)'].values)
ax14.plot(data['Lead_vy (N)'].values)
ax15.plot(data['Lead_vz (N)'].values)
plt.show()

