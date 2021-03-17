from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure
from matplotlib.pyplot import Circle
import numpy as np
import math as mt
from tkinter.filedialog import askopenfilename
from matplotlib.pylab import show
import matplotlib.animation as animation

# MEMBUAT WINDOW JENDELA DAN GRAFIK
windows = Tk()

# BANTUAN
x_p, y_p, x_z, y_z = np.zeros([2]), np.zeros([2]), np.zeros([2]), np.zeros([2])
simulasi_status = 0


class DraggableScatter():
    epsilon = 5

    def __init__(self, scatter):

        self.scatter = scatter
        self._ind = None
        self.ax = scatter.axes
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.x = None
        self.y = None
        self.canvas.draw()

    def get_ind_under_point(self, event):
        xy = np.asarray(self.scatter.get_offsets())
        xyt = self.ax.transData.transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]

        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        ind = d.argmin()

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        if event.button != 1:
            return
        self._ind = None
        pole_zero(self.x, self.y)

    def motion_notify_callback(self, event):
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.x, self.y = event.xdata, event.ydata
        if mt.sqrt(self.x ** 2 + self.y ** 2) > 1:
            self._ind = None
            return

        xy = np.asarray(self.scatter.get_offsets())
        half = int(len(xy) / 2)
        xy[self._ind] = np.array([self.x, self.y])
        if self._ind >= half:
            xy[self._ind - half] = np.array([self.x, -self.y])
        else:
            xy[self._ind + half] = np.array([self.x, -self.y])

        self.scatter.set_offsets(xy)
        self.canvas.draw_idle()


class Plot:
    def __init__(self, id, x, y, kondisi, color):
        self.id = id
        self.x = x
        self.y = y
        self.kondisi = kondisi
        self.color = color


class Axis:
    def __init__(self, judul, labelx, labely, row, column, rowspan, columnspan, tipe):
        self.labelx = labelx
        self.labely = labely
        self.row = row
        self.column = column
        self.rowspan = rowspan
        self.columnspan = columnspan
        self.title = judul
        self.tipe = tipe
        if self.tipe == 0:
            self.fig = Figure(figsize=(5, 1.1))
        else:
            self.fig = Figure(figsize=(2, 1.1))
        self.grafik_windows = FigureCanvasTkAgg(self.fig, windows)
        self.ax = self.fig.add_subplot(111)
        self.grafik_windows.get_tk_widget().grid(row=self.row,
                                                 column=self.column,
                                                 rowspan=self.rowspan,
                                                 columnspan=self.columnspan)

        self.ax.set_title(self.title, fontsize=8)
        self.ax.set_xlabel(self.labelx, fontsize=8)
        self.ax.set_ylabel(self.labely, fontsize=8)
        self.ax.tick_params(direction='in', labelsize=6)
        box = self.ax.get_position()
        if self.tipe == 0:
            self.ax.set_position([box.x0 - box.width * 0.07, box.y0 + box.height * 0.18,
                                  box.width * 1.15, box.height * 0.75])
        else:
            self.ax.set_position([box.x0 + box.width * 0.07, box.y0 + box.height * 0.18,
                                  box.width * 0.935, box.height * 0.75])
        self.line_ani = None
    def draw_plot(self):
        self.ax.clear()
        # self.__init__(self.title, self.labelx, self.labely, self.row,
        #               self.column, self.rowspan, self.columnspan, self.tipe)
        for item in self.plotlist:
            if item.kondisi == 0:
                self.ax.plot(item.x, item.y, color=item.color, linewidth=0.5)
            elif item.kondisi == 1:
                self.ax.bar(item.x, item.y, color=item.color)
        self.grafik_windows.draw()

    def plot(self, x, y, kondisi=0, color='b'):
        self.plotlist = []
        self.add_plot(0, x, y, kondisi, color)

    def add_plot(self, id, x, y, kondisi=0, color='b'):
        for i, item in enumerate(self.plotlist):  # Jika ada plot dengan id yg sama
            if item.id == id:
                self.plotlist[i] = Plot(id, x, y, kondisi, color)
                self.draw_plot()
                return

        self.plotlist += [Plot(id, x, y, kondisi, color)]  # Jika belum ada plot dengan id yang ditentukan
        self.draw_plot()

    def simulation(self, ws, x, y, d_s, color='blue'):
        self.ax.set_ylim(min(y) - 0.2, max(y) + 0.2)
        self.ax.set_xlim(0, ws)
        self.t1 = 0
        self.enum = 0
        self.t = np.arange(ws)
        self.yp = np.zeros([ws])
        self.y_sim = np.zeros([ws])
        self.p1, = self.ax.plot([], [], color=color, linewidth=0.5)
        self.p2, = self.ax.plot([], [], color=color, linewidth=0.5)
        self.grafik_windows.draw()
        self.pause = False
        self.downsampling = d_s
        b3.__setitem__('text', 'STOP')
        b3.__setitem__('command', ax15.stop)

        def onClick(event):
            self.pause ^= True

        def animasigerak(num):
            global y
            global x

            if not self.pause:
                self.yp[self.t1] = y[self.enum*self.downsampling]

                if self.t1 >= ws - 1:
                    self.t1 = 0

                if self.enum*self.downsampling >= np.size(y) - self.downsampling:
                    self.enum = 0

                self.p1.set_data(self.t[0:self.t1], self.yp[0:self.t1])
                self.p2.set_data(self.t[self.t1 + 10:ws], self.yp[self.t1 + 10:ws])

                self.y_sim = np.concatenate((self.yp[self.t1 + 1:ws - 1], self.yp[0:self.t1]))
                yBPF = BPF(self.y_sim)
                yDRV = DRV(yBPF)
                ySQR = SQR(yDRV)
                yMWI, th, LP, LN = MWI(ySQR)
                peak_logic = Peak(yMWI, th)
                RtoR = RR(peak_logic)
                Miss_3, Miss_4 = Hitung(RtoR)
                e4.delete(0, END)
                e5.delete(0, END)
                e4.insert(0, '{:5.2f}'.format(Miss_3*self.downsampling))
                e5.insert(0, '{:4.2f}'.format(Miss_4/self.downsampling))

                self.t1 += 1
                self.enum += 1

            return self.p1, self.p2

        self.line_ani = animation.FuncAnimation(self.fig, animasigerak, frames=len(y), interval=0.01, blit=True,
                                                repeat=True)
        self.fig.canvas.mpl_connect('button_press_event', onClick)

        show()

    def clear(self):
        self.ax.clear()
        self.__init__(self.title, self.labelx, self.labely, self.row,
                      self.column, self.rowspan, self.columnspan, self.tipe)

    def stop(self):
        self.line_ani.event_source.stop()
        b3.__setitem__('text', 'SIMULATE')
        b3.__setitem__('command', Simulation)


class Axis2:
    def __init__(self, judul, labelx, labely, row, column, rowspan, columnspan):
        self.labelx = labelx
        self.labely = labely
        self.row = row
        self.column = column
        self.rowspan = rowspan
        self.columnspan = columnspan
        self.title = judul
        self.fig = Figure(figsize=(2, 1.1))
        self.grafik_windows = FigureCanvasTkAgg(self.fig, windows)
        self.ax = self.fig.add_subplot(111)
        self.grafik_windows.get_tk_widget().grid(row=self.row,
                                                 column=self.column,
                                                 rowspan=self.rowspan,
                                                 columnspan=self.columnspan)
        self.dr = None
        self.ax.set_xlim((-1.05, 1.05))
        self.ax.set_ylim((-1.05, 1.05))

        self.Circle = Circle((0, 0), 1, color='black', fill=False)
        self.ax.text(-2.375, 1.1, self.title, fontsize=8)
        self.ax.text(1.1, 0, self.labelx, fontsize=8)
        self.ax.text(0, 1.1, self.labely, fontsize=8)
        self.ax.add_artist(self.Circle)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['left'].set_position(('data', 0))
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['right'].set_color(('none'))
        self.ax.spines['top'].set_color(('none'))

        box = self.ax.get_position()
        self.ax.set_position([box.x0 + box.width * 0.25, box.y0 - box.height * 0.,
                              box.width * 0.55, box.height * 1.])

    def draw_plot(self):
        self.ax.clear()
        self.__init__(self.title, self.labelx, self.labely, self.row,
                      self.column, self.rowspan, self.columnspan)
        for item in self.plotlist:
            if item.kondisi == 0:
                self.ax.scatter(item.x, item.y, color=item.color, marker='o')
            elif item.kondisi == 1:
                self.scatter = self.ax.scatter(item.x, item.y, color=item.color, marker='x')
                self.dr = DraggableScatter(self.scatter)
        self.grafik_windows.draw()

    def plot(self, x, y, kondisi=0, color='b'):
        self.plotlist = []
        self.add_plot(0, x, y, kondisi, color)

    def add_plot(self, id, x, y, kondisi=0, color='b'):
        for i, item in enumerate(self.plotlist):  # Jika ada plot dengan id yg sama
            if item.id == id:
                self.plotlist[i] = Plot(id, x, y, kondisi, color)
                self.draw_plot()
                return

        self.plotlist += [Plot(id, x, y, kondisi, color)]  # Jika belum ada plot dengan id yang ditentukan
        self.draw_plot()

    def clear(self):
        self.ax.clear()
        self.__init__(self.title, self.labelx, self.labely, self.row,
                      self.column, self.rowspan, self.columnspan)


# MENEMPELKAN AREA GRAFIK PADA WINDOWS
ax1 = Axis("Sinyal ECG", 'n', 'x[n]', 3, 1, 4, 5, 0)
ax2 = Axis("Windowing", 'n', 'x[n]', 8, 1, 4, 2, 1)
ax4 = Axis2("z-plane", r'$Re(z)$', r'$Im(z)$', 8, 4, 4, 2)
ax5 = Axis("Frekuensi Spektrum", 'Hz', 'A', 13, 1, 4, 5, 0)
ax6 = Axis("Band Pass Filter", 'n', 'x[n]', 3, 7, 4, 5, 0)
ax7 = Axis("Derivative", 'n', 'x[n]', 8, 7, 4, 5, 0)
ax8 = Axis("Square", 'n', 'x[n]', 13, 7, 4, 5, 0)
ax9 = Axis("Moving Window Integration", 'n', 'x[n]', 18, 7, 4, 5, 0)
ax10 = Axis("Level Peak", 'n', 'x[n]', 23, 7, 4, 5, 0)
ax11 = Axis("Level Noise", 'n', 'x[n]', 3, 13, 4, 5, 0)
ax12 = Axis("Threshold", 'n', 'x[n]', 8, 13, 4, 5, 0)
ax13 = Axis("Peak Logic", 'n', 'x[n]', 13, 13, 4, 5, 0)
ax14 = Axis("R to R", 'n', 'x[n]', 18, 13, 4, 5, 0)
ax15 = Axis("Simulation", 'n', 'x[n]', 18, 1, 4, 5, 0)

# JUDUL WINDOW
font9 = "-family {Showcard Gothic} -size 20 -weight bold " \
        "-slant roman -underline 0 -overstrike 0"
title = Label(windows, text="Heartrate Detection", font=font9)
title.grid(row=0, column=6, columnspan=7)


def spin_init():
    spin1.__setitem__('from_', 0)
    spin1.__setitem__('to', N)
    spin2.__setitem__('from_', 0)
    spin2.__setitem__('to', N)

def ax_init():
    global  simulasi_status
    ax1.clear()
    ax2.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax7.clear()
    ax8.clear()
    ax9.clear()
    ax10.clear()
    ax11.clear()
    ax12.clear()
    ax13.clear()
    ax14.clear()
    if simulasi_status == 1:
        ax15.stop()
        simulasi_status = 0
        ax15.clear()


def all_init():
    global n, y1, y2

    spin1.delete(0, END)
    spin2.delete(0, END)
    spin3.delete(0, END)
    spin3.insert(0, '{:3.2f}'.format(0))
    spin4.delete(0, END)
    spin4.insert(0, 0)
    spin4.__setitem__('from_', 0)
    e1.delete(0, END)
    e2.delete(0, END)
    e3.delete(0, END)
    e4.delete(0, END)
    e5.delete(0, END)
    n, y1, y2 = [], [], []
    ax_init()
    jenisnya.set(None)
    inputnya.set(None)


# input
def awal():
    global N, n, y_1, y_2, fs

    # baca file
    rep = askopenfilename(parent=windows)
    print(rep)

    f = open(rep, "r")
    input = f.readlines()

    n, y_1, y_2 = [], [], []
    for i, a in enumerate(input):
        if i == 0:
            continue
        elif i == 1:
            b = a.replace('(', '')
            c = b.replace(' sec)', '')
            d = c.split('\t')
            fs = (1 / float(d[0]))
            continue
        aa = a.replace('\n', '')
        aaa = aa.split('\t')
        n += [float(aaa[0])]
        y_1 += [float(aaa[1])]
        y_2 += [float(aaa[2])]

    N = np.size(n)
    ax_init()
    e1.delete(0, END)
    e1.insert(0, int(fs))


def pilih_input():
    global y

    y = []
    if int(inputnya.get()) == 1:
        for i in range(N):
            y += [y_1[i]]

    if int(inputnya.get()) == 2:
        for i in range(N):
            y += [y_2[i]]

    spin_init()
    ax1.plot(n, y, 0)


# Discrete Fourier Transform
def DFT(yy):
    n = np.size(yy)
    DFT_r = np.zeros([n])
    DFT_i = np.zeros([n])
    y_data = np.zeros([n])
    for i in range(n):
        for j in range(n):
            a_1 = yy[j] * mt.cos((2 * np.pi * i * j) / n)
            a_2 = -(yy[j] * mt.sin((2 * np.pi * i * j) / n))
            DFT_r[i] += a_1
            DFT_i[i] += a_2
        y_data[i] = (mt.sqrt((DFT_r[i] ** 2) + (DFT_i[i] ** 2))) / n
    return y_data


def Hanning():
    global yHann
    s1 = int(spin1.get())
    s2 = int(spin2.get())
    index = 0

    yHann = np.zeros([N])
    for i in range(s1, s2):
        yHann[i] = (0.5 - (0.5 * mt.cos((2 * np.pi * index) / ((s2 - s1) - 1))))
        index += 1
    ax1.add_plot(4, n, yHann, color='red')
    ax1.add_plot(5, [s1, s1], [0, max(y)], 0, 'red')
    ax1.add_plot(6, [s2, s2], [0, max(y)], 0, 'red')
    ax2.plot(n[s1:s2], y[s1:s2])


def Windowing():
    global yWindow

    yWindow = np.multiply(y, yHann)
    yDFT = DFT(yWindow)
    if np.amax(yDFT) != 0:
        yDFT = yDFT / np.amax(yDFT)  # agar tidak terjadi gain

    ax5.plot(np.arange(int(np.size(yDFT)))/N,
             yDFT[0:int(np.size(yDFT))], 0)


def pole_zero(xp=x_p[0], yp=y_p[0]):
    global s_p, x_p, y_p, x_z, y_z

    if xp == 0:
        s_p = 0
    else:
        s_p = float(360 * mt.atan(yp / xp) / np.pi)

    r_p = mt.sqrt(xp ** 2 + yp ** 2)
    spin3.delete(0, END)
    spin4.delete(0, END)
    spin3.insert(0, '{:3.2f}'.format(r_p))
    spin4.insert(0, int(s_p))
    if (int(jenisnya.get())) == 1:
        x_z = [-1, -1]
        y_z = [0, 0]

    if (int(jenisnya.get())) == 2:
        x_z = [1, 1]
        y_z = [0, 0]

    if (int(jenisnya.get())) == 3:
        x_z = [-1, 1]
        y_z = [0, 0]

    if (int(jenisnya.get())) == 4:
        if xp < 0:
            x_z = [-mt.cos((s_p * np.pi) / 360), -mt.cos((s_p * np.pi) / 360)]
        else:
            x_z = [mt.cos((s_p * np.pi) / 360), mt.cos((s_p * np.pi) / 360)]
        y_z = [mt.sin((s_p * np.pi) / 360), -mt.sin((s_p * np.pi) / 360)]

    x_p = [xp, xp]
    y_p = [yp, -yp]

    f_resp = []
    for i in range(int(N)):
        w = (2 * np.pi * i) / N
        a_1 = x_p[0] + x_p[1]
        a_2 = y_p[0] + y_p[1]
        a_3 = ((x_p[0]*x_p[1])-(y_p[0]*y_p[1]))
        a_4 = ((x_p[0]*y_p[1])+(y_p[0]*x_p[1]))
        b_1 = x_z[0] + x_z[1]
        b_2 = y_z[0] + y_z[1]
        b_3 = ((x_z[0] * x_z[1]) - (y_z[0] * y_z[1]))
        b_4 = ((x_z[0] * y_z[1]) + (y_z[0] * x_z[1]))
        num = mt.sqrt((1 - b_1*mt.cos(w) - b_2*mt.sin(w) + b_3*mt.cos(2*w) + b_4*mt.sin(2*w))**2
                        + (b_1*mt.sin(w) - b_2*mt.cos(w) - b_3*mt.sin(2*w) + b_4*mt.cos(2*w))**2)
        denum = mt.sqrt((1 - a_1*mt.cos(w) - a_2*mt.sin(w) + a_3*mt.cos(2*w) + a_4*mt.sin(2*w))**2
                        + (a_1*mt.sin(w) - a_2*mt.cos(w) - a_3*mt.sin(2*w) + a_4*mt.cos(2*w))**2)
        f_resp += [num / denum]

    if np.amax(f_resp) != 0:
        f_resp = f_resp / np.amax(f_resp)  # agar tidak terjadi gain

    ax4.plot(x_z, y_z)
    ax4.add_plot(1, x_p, y_p, 1)
    ax5.add_plot(1, np.arange(N)/N, f_resp)


# BPF
def BPF(non_filtered):
    yBPF = np.zeros(np.size(non_filtered))
    for i in range(np.size(non_filtered)):
        yBPF[i] += sum(x_p) * yBPF[i - 1] - (x_p[0] * x_p[1] - y_p[0] * y_p[1]) * yBPF[i - 2] + non_filtered[i]
        if i > 1: yBPF[i] += -sum(x_z) * non_filtered[i - 1]
        if i > 2: yBPF[i] += (x_z[0] * x_z[1] - y_z[0] * y_z[1]) * non_filtered[i - 2]

    if np.amax(yBPF) != 0:
        yBPF = yBPF / np.amax(yBPF)  # agar tidak terjadi gain

    return yBPF


def DRV(yBPF):
    yDRV = []
    for i in range(np.size(yBPF)):
        if i < 2:
            a = 0
        else:
            a = -yBPF[i - 2]
        if i < 1:
            b = 0
        else:
            b = -2 * yBPF[i - 1]
        if i > np.size(yBPF) - 2:
            c = 0
        else:
            c = 2 * yBPF[i + 1]
        if i > np.size(yBPF) - 3:
            d = 0
        else:
            d = yBPF[i + 2]
        yDRV += [(a + b + c + d) / 8]

    if np.amax(yDRV) != 0:
        yDRV = yDRV / np.amax(yDRV)  # agar tidak terjadi gain

    return yDRV


# Squaring
def SQR(yDRV):
    ySQR = list(map(lambda x: x ** 2, yDRV))

    return ySQR


# Moving Window Integration
def MWI(ySQR):
    yMWI = []
    Maks = np.zeros([np.size(ySQR)])
    WSI = 60
    λp = 0.98
    λn = 0.98
    τ = 0.4
    for i in range(np.size(ySQR)):
        MAV = 0
        for j in range(WSI):
            if i - (WSI - j) >= 0:
                MAV += ySQR[i - (WSI - j)]
        yMWI += [MAV / WSI]

    for i in range(np.size(yMWI)):
        Maks[i] = yMWI[0]
        if yMWI[i] >= Maks[i]:
            Maks[i] = yMWI[i]

    LP, LN, th = np.zeros([np.size(yMWI)]), np.zeros([np.size(yMWI)]), np.zeros([np.size(yMWI)])
    for i in range(np.size(yMWI)):
        if i > 0:
            LP[i] = λp * LP[i - 1] + (1-λp) * Maks[i]
            LN[i] = λn * LN[i - 1] + (1-λp) * 0.1 * Maks[i]
            th[i] = (LN[i] + τ * (LP[i] - LN[i]))
        else:
            LP[i] = λp * LP[i - 1]
            LN[i] = λn * LN[i - 1]
            th[i] = (LN[i] + τ * (LP[i] - LN[i]))
    return yMWI, th, LP, LN


def Peak(yMWI, th):  # menentukan peak detector
    peak_logic = np.zeros([np.size(yMWI)])

    for i in range(len(yMWI)):
        if yMWI[i] < th[i]:
            peak_logic[i] = 0
        elif yMWI[i] >= th[i]:
            peak_logic[i] = 1

    return peak_logic


def RR(peak_logic):
    R = []
    RtoR = []

    status = 1
    for i in range(np.size(peak_logic)):
        if status == 0:
            if peak_logic[i] != 0:
                R += [i]
                status = 1
        if peak_logic[i] == 0:
            status = 0

    if np.size(R) <= 1:
        return [0]
    else:
        for i in range(int(np.size(R) - 1)):
            RtoR += [R[i + 1] - R[i]]
        return RtoR


def Hitung(RtoR):
    Miss_1 = np.mean(RtoR)
    if Miss_1 == 0:
        Miss_2 = 0
    else:
        Miss_2 = (fs * 60) / Miss_1
    return Miss_1, Miss_2


def Generate():
    global Miss_1, Miss_2

    yBPF = BPF(y)
    ax6.plot(n, yBPF)

    yDRV = DRV(yBPF)
    ax7.plot(n, yDRV)

    ySQR = SQR(yDRV)
    ax8.plot(n, ySQR)

    yMWI, th, LP, LN = MWI(ySQR)
    ax9.plot(n, yMWI)
    ax10.plot(n, yMWI)
    ax10.add_plot(1, n, LP, color='red')
    ax11.plot(n, yMWI)
    ax11.add_plot(2, n, LN, color='red')
    ax12.plot(n, yMWI)
    ax12.add_plot(3, n, th, color='red')

    peak_logic = Peak(yMWI, th)
    ax13.plot(n, peak_logic)

    RtoR = RR(peak_logic)
    ax14.plot(np.arange(np.size(RtoR)), RtoR, kondisi=1)

    Miss_1, Miss_2 = Hitung(RtoR)


def Result_Show():
    e2.delete(0, END)
    e3.delete(0, END)
    e2.insert(0, '{:5.2f}'.format(Miss_1))
    e3.insert(0, '{:4.2f}'.format(Miss_2))


def Simulation():
    global simulasi_status

    downsampling = 1
    if fs>150 and fs<300 : downsampling = 2
    if fs>300 : downsampling = 3
    simulasi_status = 1
    ws = int((4/downsampling) * fs)
    ax15.simulation(ws, n, y, downsampling)

def Quit():
##    windows.quit()     # stops mainloop
    windows.destroy()

# DEFINISIKAN LABEL PADA WINDOW
l1 = Label(windows, text='Fs\t=')
l1.grid(row=2, column=1)
l2 = Label(windows, text='R to R =')
l2.grid(row=23, column=2)
l3 = Label(windows, text='Heartrate =')
l3.grid(row=24, column=2)
l4 = Label(windows, text='R to R =')
l4.grid(row=23, column=14)
l44 = Label(windows, text='Heartrate =')
l44.grid(row=24, column=14)
l5 = Label(windows, text='↓')
l5.grid(row=22, column=9)
l6 = Label(windows, text='↓')
l6.grid(row=17, column=9)
l7 = Label(windows, text='↓')
l7.grid(row=12, column=9)
l8 = Label(windows, text='↓')
l8.grid(row=7, column=9)
l9 = Label(windows, text=' ', fg="white")
l9.grid(row=2, column=0, rowspan=25)
l10 = Label(windows, text=' ')
l10.grid(row=2, column=6, rowspan=25)
l11 = Label(windows, text=' ')
l11.grid(row=2, column=12, rowspan=25)
l12 = Label(windows, text='Start :')
l12.grid(row=7, column=1, sticky=W)
l12 = Label(windows, text='Stop :')
l12.grid(row=7, column=2, sticky=W)
l13 = Label(windows, text='R  :')
l13.grid(row=7, column=4, sticky=W)
l14 = Label(windows, text='φ  :')
l14.grid(row=7, column=5, sticky=W)
l15 = Label(windows, text=' ')
l15.grid(row=31, column=9)
l16 = Label(windows, text=' ')
l16.grid(row=31, column=18)

# DEFINISI INPUT TEXT STRING
e1 = Entry(windows, width=8)
e1.grid(row=2, column=2)
e2 = Entry(windows, width=8)
e2.grid(row=23, column=15)
e3 = Entry(windows, width=8)
e3.grid(row=24, column=15)
e4 = Entry(windows, width=8)
e4.grid(row=23, column=3)
e5 = Entry(windows, width=8)
e5.grid(row=24, column=3)

# DEFINISI INPUT BUTTON
b1 = Button(windows, text='INPUT', command=awal)
b1.grid(row=2, column=3)
b2 = Button(windows, text='PLOT DFT', command=Windowing)
b2.grid(row=12, column=1, columnspan=2)
b3 = Button(windows, text='SIMULATE', command=Simulation, width=8)
b3.grid(row=23, column=4, rowspan=2)
b4 = Button(windows, text='GENERATE', command=Generate)
b4.grid(row=2, column=9)
b5 = Button(windows, text='RESULT', command=Result_Show)
b5.grid(row=23, column=16, rowspan=2)
b6 = Button(windows, text='RESTART', command=all_init)
b6.grid(row=2, column=14)
b7 = Button(windows, text='QUIT', command=Quit)
b7.grid(row=2, column=16)

# SPINBOX
spin1 = Spinbox(windows, from_=0, to=None, width=4, command=Hanning)
spin1.grid(row=7, column=1)
spin2 = Spinbox(windows, from_=0, to=None, width=4, command=Hanning)
spin2.grid(row=7, column=2)
spin3 = Spinbox(windows, from_=0, to=1, width=4, format="%.2f", increment=0.01, command=pole_zero)
spin3.grid(row=7, column=4)
spin4 = Spinbox(windows, from_=0, to=360, width=4, command=pole_zero)
spin4.grid(row=7, column=5)

# RADIOBUTTON
inputnya = IntVar(windows)
r1 = Radiobutton(windows, text='MLII', value=1, variable=inputnya, command=pilih_input)
r1.grid(row=2, column=4)
r2 = Radiobutton(windows, text='V', value=2, variable=inputnya, command=pilih_input)
r2.grid(row=2, column=5)

jenisnya = IntVar(windows)
r3 = Radiobutton(windows, text='LPF', value=1, variable=jenisnya, command=pole_zero)
r3.grid(row=8, column=3)
r4 = Radiobutton(windows, text='HPF', value=2, variable=jenisnya, command=pole_zero)
r4.grid(row=9, column=3)
r5 = Radiobutton(windows, text='BPF', value=3, variable=jenisnya, command=pole_zero)
r5.grid(row=10, column=3)
r6 = Radiobutton(windows, text='BSF', value=4, variable=jenisnya, command=pole_zero)
r6.grid(row=11, column=3)

windows.mainloop()
