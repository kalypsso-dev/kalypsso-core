import numpy as np
import matplotlib.pyplot as plt

x = np.load("x_2d.npy")
y = np.load("y_2d.npy")

treeids = np.arange(x.size)

ax = plt.figure().add_subplot()

ax.plot(x+0.5, y+0.5)

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

#ax.axis("equal")
#ax.set_aspect("equal", "box")
ax.set_aspect("equal")

# Major ticks every 1
x_major_ticks = np.arange(x_min, x_max+1.5, 1)
y_major_ticks = np.arange(y_min, y_max+1.5, 1)

ax.set_xticks(x_major_ticks)
ax.set_yticks(y_major_ticks)


ax.grid()

for treeid in treeids:
    ax.annotate(treeid, xy = (x[treeid]+0.3, y[treeid]+0.3))

ax.set_xlabel('x')
ax.set_ylabel('y')

plt.title("p4est tree numbering in 2d brick connectivity.")

plt.show()
