from matplotlib import pyplot as plt
import numpy as np

mean_4 = np.load("4m_mean.npy")
std_4 = np.load("4m_std.npy")
mean_7 = np.load("7m_mean.npy")
std_7 = np.load("7m_std.npy")

index = np.arange(mean_7.shape[0])
np.random.shuffle(index)
index = index[:100]
mean_4 = mean_4[index]
std_4 = std_4[index]
mean_7 = mean_7[index]
std_7 = std_7[index]

plt.subplot(1, 2, 1)
plt.title("Mean")
plt.hist(mean_4.reshape(-1), label="4m")
plt.hist(mean_7.reshape(-1), label="7m")
plt.legend()
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.title("Variance")
plt.hist(std_4.reshape(-1), label="4m")
plt.hist(std_7.reshape(-1), label="7m")
plt.legend()
plt.tight_layout()
plt.savefig("distribution.png", dpi=600)
