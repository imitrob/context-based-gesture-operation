import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_circle_gradient(inner_radius = 1, outer_radius = 3, center = [6,4], color = 'gold'):
    center_x, center_y = center
    halo_color = color
    center_color = color

    xmin = center_x - outer_radius
    xmax = center_x + outer_radius
    ymin = center_y - outer_radius
    ymax = center_y + outer_radius
    x, y = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    z = np.where(r < inner_radius, np.nan, np.clip(outer_radius - r, 0, np.inf))
    cmap = LinearSegmentedColormap.from_list('', ['#FFFFFF00', halo_color])
    cmap.set_bad(center_color)
    plt.imshow(z, cmap=cmap, extent=[xmin, xmax, ymin, ymax], origin='lower', zorder=3)
    plt.axis('equal')


if __name__ == '__main__':
    plot_circle_gradient(inner_radius = 3, outer_radius = 4, center = [4,4], color='blue')
    plot_circle_gradient(inner_radius = 3, outer_radius = 4, center = [8,4], color='yellow')
    plt.show()
