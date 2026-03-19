import numpy as np
import matplotlib.pyplot as plt

x = np.load("x_3d.npy")
y = np.load("y_3d.npy")
z = np.load("z_3d.npy")

treeids = np.arange(x.size)

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)
z_min = np.min(z)
z_max = np.max(z)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x+0.5, y+0.5, z+0.5)

# Major ticks every 1
x_major_ticks = np.arange(x_min, x_max+1.5, 1)
y_major_ticks = np.arange(y_min, y_max+1.5, 1)
z_major_ticks = np.arange(z_min, z_max+1.5, 1)

ax.set_xticks(x_major_ticks)
ax.set_yticks(y_major_ticks)
ax.set_zticks(z_major_ticks)
ax.grid()

for treeid in treeids:
    ax.text(x[treeid]+0.4, y[treeid]+0.4, z[treeid]+0.4, '%s' % (str(treeid)),size=16,color='r')

#plt.axis("equal")

plt.title("p4est tree numbering in 3d brick connectivity.")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
