from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

class InteractivePlot():
    def __init__(self):
        self.axis_color = 'lightgoldenrodyellow'
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # Adjust the subplots region to leave some space for the sliders and buttons
        self.fig.subplots_adjust(left=0.25, bottom=0.25)

        self.t = np.arange(0.0, 1.0, 0.001)
        self.amp_0 = 5
        self.freq_0 = 3

        # Draw the initial plot
        # The 'line' variable is used for modifying the line later
        [self.line] = self.ax.plot(self.t, self.signal(self.amp_0, self.freq_0), linewidth=2, color='red')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([-10, 10])

        # Define an axes area and draw a slider in it
        self.amp_slider_ax  = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=self.axis_color)
        self.amp_slider = Slider(self.amp_slider_ax, 'Amp', 0.1, 10.0, valinit=self.amp_0)

        # Draw another slider
        self.freq_slider_ax = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=self.axis_color)
        self.freq_slider = Slider(self.freq_slider_ax, 'Freq', 0.1, 30.0, valinit=self.freq_0)

        self.amp_slider.on_changed(self.sliders_on_changed)
        self.freq_slider.on_changed(self.sliders_on_changed)

        # Add a button for resetting the parameters
        self.reset_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_ax, 'Reset', color=self.axis_color, hovercolor='0.975')

        self.reset_button.on_clicked(self.reset_button_on_clicked)

        # Add a set of radio buttons for changing color
        self.color_radios_ax = self.fig.add_axes([0.025, 0.5, 0.15, 0.15], facecolor=self.axis_color)
        self.color_radios = RadioButtons(self.color_radios_ax, ('red', 'blue', 'green'), active=0)

        self.color_radios.on_clicked(self.color_radios_on_clicked)

        plt.show()

    def color_radios_on_clicked(self, label):
        self.line.set_color(label)
        self.fig.canvas.draw_idle()

    def reset_button_on_clicked(self, mouse_event):
        self.freq_slider.reset()
        self.amp_slider.reset()

    def signal(self, amp, freq):
        return amp * sin(2 * pi * freq * self.t)

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(self, val):
        self.line.set_ydata(signal(self.amp_slider.val, self.freq_slider.val))
        self.fig.canvas.draw_idle()


ip = InteractivePlot()



#
