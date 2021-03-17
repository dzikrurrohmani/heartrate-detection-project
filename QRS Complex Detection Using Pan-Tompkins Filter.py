from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure
import numpy as np
import math as mt
from tkinter.filedialog import askopenfilename
from matplotlib.pylab import show
import matplotlib.animation as animation

# MEMBUAT WINDOW JENDELA DAN GRAFIK
windows = Tk()

# BANTUAN
simulasi_status = 0

class Plot:
    def __init__(self, id, x, y, kondisi, color):
        self.id = id
        self.x = x
        self.y = y
        self.kondisi = kondisi
        self.color = color


class Axis:
    def __init__(self, judul, labelx, labely, row, column, rowspan, columnspan):
        self.labelx = labelx
        self.labely = labely
        self.row = row
        self.column = column
        self.rowspan = rowspan
        self.columnspan = columnspan
        self.title = judul
        self.fig = Figure(figsize=(6, 1.1))
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
        # self.ax.grid(True)
        box = self.ax.get_position()
        self.ax.set_position([box.x0 - box.width * 0.07, box.y0 + box.height * 0.18,
                              box.width * 1.15, box.height * 0.75])

    def draw_plot(self):
        self.ax.clear()
        self.__init__(self.title, self.labelx, self.labely, self.row,
                      self.column, self.rowspan, self.columnspan)
        for item in self.plotlist:
            if item.kondisi == 0:
                self.ax.plot(item.x, item.y, color=item.color, linewidth=0.5)
            elif item.kondisi == 1:
                self.ax.bar(item.x, item.y, color=item.color, linewidth=0.5)
        self.grafik_windows.draw()

    def plot(self, x, y, kondisi=0, color='blue'):
        self.plotlist = []
        self.add_plot(0, x, y, kondisi, color)

    def add_plot(self, id, x, y, kondisi=0, color='blue'):
        for i, item in enumerate(self.plotlist):  # Jika ada plot dengan id yg sama
            if item.id == id:
                self.plotlist[i] = Plot(id, x, y, kondisi, color)
                self.draw_plot()
                return

        self.plotlist += [Plot(id, x, y, kondisi, color)]  # Jika belum ada plot dengan id yang ditentukan
        self.draw_plot()

    def clear(self):
        self.ax.clear()

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
        b4.__setitem__('text', 'STOP')
        b4.__setitem__('command', ax9.stop)

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
                yMWI, th, LP, LN = MVI(ySQR)
                peak_logic = Peak(yMWI, th)
                RtoR = RR(peak_logic)
                Miss_3, Miss_4 = Hitung(RtoR)
                e4.delete(0, END)
                e5.delete(0, END)
                e4.insert(0, Miss_3*self.downsampling)
                e5.insert(0, Miss_4/self.downsampling)

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
                      self.column, self.rowspan, self.columnspan)

    def stop(self):
        self.line_ani.event_source.stop()
        b4.__setitem__('text', 'SIMULATE')
        b4.__setitem__('command', Simulation)

# MENEMPELKAN AREA GRAFIK PADA WINDOWS
ax1 = Axis("Sinyal ECG", 'n', 'x[n]'                    , 3, 1, 4, 5)
ax2 = Axis("Frekuensi Spectrum", 'Hz', 'A'              , 8, 1, 4, 5)
ax3 = Axis("Band Pass Filter", 'n', 'x[n]'              , 13, 1, 4, 5)
ax4 = Axis("Frekuensi Spectrum", 'Hz', 'A'              , 18, 1, 4, 5)
ax5 = Axis("Derivative", 'n', 'x[n]'                    , 23, 1, 4, 5)
ax6 = Axis("SQUARE", 'n', 'x[n]'                        , 3, 7, 4, 5)
ax7 = Axis("Moving Window Integration", 'n', 'x[n]'     , 8, 7, 4, 5)
ax8 = Axis("Peak Logic", 'n', 'x[n]'                    , 13, 7, 4, 5)
ax9 = Axis("Simulation", 'n', 'x[n]'                    , 18, 7, 4, 5)

# JUDUL WINDOW
font9 = "-family {Showcard Gothic} -size 20 -weight bold " \
        "-slant roman -underline 0 -overstrike 0"
title = Label(windows, text="Heartrate Detection", font=font9)
title.grid(row=0, column=3, columnspan=7)

def ax_init():
    global simulasi_status

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax7.clear()
    ax8.clear()
    if simulasi_status == 1:
        ax9.stop()
        simulasi_status = 0
        ax9.clear()

def all_init():
    global n, y_1, y_2

    spin1.delete(0, END)
    e1.delete(0, END)
    e2.delete(0, END)
    e3.delete(0, END)
    e4.delete(0, END)
    e5.delete(0, END)
    n, y_1, y_2 = [], [], []
    ax_init()
    inputnya.set(None)

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

# LPF
def BPF(y):
    yLPF = []
    for i in range(np.size(y)):
        if i<1: a = 0
        else : a = 2*yLPF[i-1]
        if i<2: b = 0
        else: b = -yLPF[i-2]
        c = y[i]
        if i<6: d = 0
        else: d = -2*y[i-6]
        if i<12: e = 0
        else: e = y[i-12]
        yLPF += [a+b+c+d+e]

    if np.amax(yLPF)>1:
        yLPF = yLPF / np.amax(yLPF)  # agar tidak terjadi gain

    yHPF = []
    for i in range(np.size(yLPF)):
        if i<16: a = 0
        else : a = 32*yLPF[i-16]
        if i<1: b = 0
        else: b = yHPF[i-1]
        c = yLPF[i]
        if i<32: d = 0
        else: d = -yLPF[i-32]
        yHPF += [a-(b+c+d)]

    if np.amax(yHPF) > 1:
        yHPF = yHPF / np.amax(yHPF)
    return yHPF

# Derivative
def DRV(yHPF):
    yDRV = []
    for i in range (np.size(yHPF)):
        if i<2: a = 0
        else: a = -yHPF[i-2]
        if i<1: b = 0
        else: b = -2*yHPF[i-1]
        if i>np.size(yHPF)-2: c = 0
        else:c = 2*yHPF[i+1]
        if i>np.size(yHPF)-3: d=0
        else: d = yHPF[i+2]
        yDRV += [(a+b+c+d)/8]

    if np.amax(yDRV) > 1:
        yDRV = yDRV/np.amax(yDRV)

    return yDRV

def SQR(yDRV):
    ySQR = list(map(lambda x: x ** 2, yDRV))
    return ySQR

# Moving Window Integration
def MVI(ySQR):
    yMWI = []
    Maks = np.zeros([np.size(ySQR)])
    WSI = int(spin1.get())
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
    return  Miss_1, Miss_2

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
    e1.insert(0, fs)

def pilih_input():
    global y

    y = []
    if int(inputnya.get()) == 1:
        for i in range(N):
            y += [y_1[i]]

    if int(inputnya.get()) == 2:
        for i in range(N):
            y += [y_2[i]]

    ax1.plot(n, y, 0)

def Generate():
    global ySQR

    yDFT1 = DFT(y)
    ax2.plot(np.arange(int(np.size(yDFT1) / 2)), yDFT1[0:int(np.size(yDFT1) / 2)], 1)

    yBPF = BPF(y)
    yDFT2 = DFT(yBPF)
    ax3.plot(n, yBPF, 0)
    ax4.plot(np.arange(int(np.size(yDFT2) / 2)), yDFT2[0:int(np.size(yDFT2) / 2)], 1)

    yDRV = DRV(yBPF)
    ax5.plot(n, yDRV, 0)

    ySQR = SQR(yDRV)
    ax6.plot(n, ySQR, 0)

    spin1.__setitem__('from_', 0 )
    spin1.__setitem__('to', N)

def Generate2():
    global Miss_1, Miss_2

    yMWI, th, LP, LN = MVI(ySQR)
    ax7.plot(n, yMWI)
    ax7.add_plot(1, n, th, color='red')
    ax7.add_plot(2, n, LP, color='red')
    ax7.add_plot(3, n, LN, color='red')

    peak_logic = Peak(yMWI, th)
    ax8.plot(n, peak_logic)

    RtoR = RR(peak_logic)
    Miss_1, Miss_2 = Hitung(RtoR)

def Result_Show():
    e2.delete(0, END)
    e3.delete(0, END)
    e2.insert(0, Miss_1)
    e3.insert(0, Miss_2)

def Simulation():
    global simulasi_status

    downsampling = 1
    if fs>150 and fs<300 : downsampling = 2
    if fs>300 : downsampling = 3
    simulasi_status = 1
    ws = int((4/downsampling) * fs)
    ax9.simulation(ws, n, y, downsampling)

# DEFINISIKAN LABEL PADA WINDOW
l1 = Label(windows, text='Fs\t=')
l1.grid(row=2, column=1)
l2 = Label(windows, text='R to R =')
l2.grid(row=25, column=7)
l3 = Label(windows, text='Heartrate =')
l3.grid(row=26, column=7)
l4 = Label(windows, text='R to R =')
l4.grid(row=25, column=10)
l5 = Label(windows, text='Heartrate =')
l5.grid(row=26, column=10)
l6 = Label(windows, text='Window Integration Size =')
l6.grid(row=7, column=7, columnspan=2)
l7 = Label(windows, text=' ')
l7.grid(row=2, column=0)
l8 = Label(windows, text=' ')
l8.grid(row=27, column=12)
l9 = Label(windows, text=' ')
l9.grid(row=12, column=6)
l10 = Label(windows, text=' ')
l10.grid(row=17, column=6)
l11 = Label(windows, text=' ')
l11.grid(row=22, column=6)

# DEFINISI INPUT TEXT STRING
e1 = Entry(windows, width=8)
e1.grid(row=2, column=2)
e2 = Entry(windows, width=8)
e2.grid(row=25, column=8)
e3 = Entry(windows, width=8)
e3.grid(row=26, column=8)
e4 = Entry(windows, width=8)
e4.grid(row=25, column=11)
e5 = Entry(windows, width=8)
e5.grid(row=26, column=11)

# DEFINISI INPUT BUTTON
b1 = Button(windows, text='INPUT', command=awal)
b1.grid(row=2, column=3)
b2 = Button(windows, text='GENERATE', command=Generate)
b2.grid(row=7, column=3)
b3 = Button(windows, text='RESULT', command=Result_Show)
b3.grid(row=23, column=7, columnspan=2)
b4 = Button(windows, text='SIMULATE', command=Simulation, width=8)
b4.grid(row=23, column=10, columnspan=2)
b5 = Button(windows, text='RESTART', command=all_init)
b5.grid(row=2, column=9)

# SPINBOX
spin1 = Spinbox(windows, from_=0, to=None, width=5, command=Generate2)
spin1.grid(row=7, column=9)

# RADIOBUTTON
inputnya = IntVar(windows)
r1 = Radiobutton(windows, text='MLII', value=1, variable=inputnya, command=pilih_input)
r1.grid(row=2, column=4)
r2 = Radiobutton(windows, text='V', value=2, variable=inputnya, command=pilih_input)
r2.grid(row=2, column=5)

windows.mainloop()
