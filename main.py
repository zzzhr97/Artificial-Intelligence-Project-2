import tkinter as tk
import numpy as np
import math
from PIL import Image, ImageTk
import random
import algorithm


class Gomoku(tk.Frame):
    def __init__(self, master=None):
        super(Gomoku, self).__init__(master=master)
        sw = master.winfo_screenwidth()
        sh = master.winfo_screenheight()
        self.w = 650
        self.h = 700
        self.block = 42
        master.geometry(f'{self.w}x{self.h}+{int((sw-self.w)/2)}+{int((sh-self.h)/3)}')
        master.title('Gomoku')
        master.iconphoto(False, tk.PhotoImage(file='img/favicon.png'))
        master.resizable(False, False)

        self.left = 31
        self.top = 81
        self.start = 0
        self.last_drop = (0, 0)

        # black -1, white 1
        self.color = tk.IntVar(value=-1)
        self.color.trace('w', self.__robot)

        # empty 0, black -1, white 1
        self.chessboard = np.zeros((15, 15))

        self.canvas = tk.Canvas(master=master, width=self.w, height=self.h)

        master.one = self.bg = ImageTk.PhotoImage(image=Image.open('img/bg.png'))

        self.canvas.create_image(0, 0, image=self.bg, anchor='se')
        self.canvas.create_image(0, self.h, image=self.bg, anchor='se')
        self.canvas.create_image(self.w, 0, image=self.bg, anchor='se')
        self.canvas.create_image(self.w, self.h, image=self.bg, anchor='se')

        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.__draw_piece)
        self.canvas.bind("<Button-2>", self.__draw_piece)
        self.canvas.bind("<Button-3>", self.__draw_piece)

        self.__draw_lines()
        self.__draw_dots()
        self.__pieces = []
        self.red = None

        self.btn_start = tk.Button(master=self.canvas, text='Start', width=8,
                                   height=1, font=('Simhei', 18), command=self.__start)
        self.btn_reset = tk.Button(master=self.canvas, text='Reset', width=8, state=tk.DISABLED,
                                   height=1, font=('Simhei', 18), command=self.__reset)
        self.label = tk.Label(master=self.canvas, text='', font=('Simhei', 20), width=18, height=1)
        self.canvas.create_window(80, 30, window=self.btn_start)
        self.canvas.create_window(220, 30, window=self.btn_reset)
        self.canvas.create_window(500, 30, window=self.label)

    def __draw_lines(self):
        for i in range(15):
            w = 2
            if i == 0 or i == 14:
                w += 2

            start = (self.left+i*self.block, self.top)
            end = (self.left+i*self.block, self.top+self.block*14+2)
            self.canvas.create_line(start[0], start[1], end[0], end[1], fill='black', width=w)

            start = (self.left-2, self.top+i*self.block)
            end = (self.left+self.block*14+2, self.top+i*self.block)
            self.canvas.create_line(start[0], start[1], end[0], end[1], fill='black', width=w)

    def __draw_dots(self):
        left, top = self.left+self.block*3, self.top+self.block*3
        for i in range(3):
            for j in range(3):
                center = (left+i*self.block*4, top+j*self.block*4)
                self.canvas.create_oval(center[0]-6, center[1]-6, center[0]+6, center[1]+6, fill='black')

    def __draw_piece(self, event):
        if self.start:
            r, c = 0, 0
            flag = False
            if isinstance(event, tuple):
                r, c = event
                flag = True
                x, y = r*self.block+self.left, c*self.block+self.top
            else:
                pos = (event.x, event.y)
                r, c = int((pos[0]-self.left+self.block/2)/self.block), int((pos[1]-self.top+self.block/2)/self.block)
                x, y = r*self.block+self.left, c*self.block+self.top
                flag = math.sqrt((pos[0]-x)**2+(pos[1]-y)**2) < 15
            piece_color = {-1: 'black', 1: 'white'}
            if self.chessboard[r][c] == 0 and flag:
                if self.red:
                    self.canvas.delete(self.red)
                self.red = self.canvas.create_oval(
                    x-17, y-17, x+17, y+17, fill='red')
                self.__pieces.append(self.canvas.create_oval(
                    x-15, y-15, x+15, y+15, fill=piece_color[self.color.get()]))
                self.chessboard[r][c] = self.color.get()
                self.last_drop = (r, c)
                if self.__is_win((r, c)):
                    self.start = 0
                    self.label['text'] = 'WHITE WON!!!' if self.color.get() == 1 else 'BLACK WON!!!'
                self.update()
                self.color.set(-self.color.get())

    def __start(self):
        self.start = 1
        self.btn_start['state'] = tk.DISABLED
        self.btn_reset['state'] = tk.NORMAL
        self.player_color = -1 if random.random() < 0.5 else 1
        self.label['text'] = 'Player: White' if self.player_color == 1 else 'Player: Black'
        if self.player_color == 1:
            self.__robot()

    def __reset(self):
        self.start = 0
        self.chessboard = np.zeros((15, 15))
        self.btn_start['state'] = tk.NORMAL
        self.btn_reset['state'] = tk.DISABLED
        self.color.set(-1)
        for p in self.__pieces:
            self.canvas.delete(p)
        self.canvas.delete(self.red)
        self.red = None
        self.label['text'] = ''

    def __robot(self, *args):
        if self.player_color != self.color.get() and self.start:
            self.__draw_piece(algorithm.robot(self.chessboard, self.color.get(), self.last_drop))

    def __is_win(self, pos):
        if self.__horizontal(pos):
            return True
        if self.__vertical(pos):
            return True
        if self.__main_diag(pos):
            return True
        if self.__cont_diag(pos):
            return True
        return False

    def __horizontal(self, pos):
        r, c = pos
        left = c-4 if c-4 > 0 else 0
        right = c+4 if c+4 < 15 else 14
        check = []
        for i in range(left, right-3):
            check.append(int(abs(sum(self.chessboard[r, i:i+5])) == 5))
        return sum(check) == 1

    def __vertical(self, pos):
        r, c = pos
        top = r-4 if r-4 > 0 else 0
        bottom = r+4 if r+4 < 15 else 14
        check = []
        for i in range(top, bottom-3):
            check.append(int(abs(sum(self.chessboard[i:i+5, c])) == 5))
        return sum(check) == 1

    def __main_diag(self, pos):
        r, c = pos
        left, top, right, bottom = 0, 0, 0, 0
        if r >= c:
            left = c-4 if c-4 > 0 else 0
            bottom = r+4 if r+4 < 15 else 14
            right = bottom - (r-c)
            top = left + r-c
        else:
            right = c+4 if c+4 < 15 else 14
            top = r-4 if r-4 > 0 else 0
            left = top+c-r
            bottom = right-(c-r)

        check = []
        if right-left > 3:
            for i in range(right-left-3):
                col = np.arange(left+i, left+i+5)
                row = np.arange(top+i, top+i+5)
                check.append(int(abs(sum(self.chessboard[row, col])) == 5))
        return sum(check) == 1

    def __cont_diag(self, pos):
        r, c = pos
        left, top, right, bottom = 0, 0, 0, 0
        if r + c <= 14:
            top = r-4 if r-4 > 0 else 0
            left = c-4 if c-4 > 0 else 0
            bottom = r+c-left
            right = r+c-top
        else:
            bottom = r+4 if r+4 < 15 else 14
            right = c+4 if c+4 < 15 else 14
            top = r+c-right
            left = r+c-bottom

        check = []
        if right-left > 3:
            for i in range(right-left-3):
                col = np.arange(left+i, left+i+5)
                row = np.arange(bottom-i, bottom-i-5, -1)
                check.append(int(abs(sum(self.chessboard[row, col])) == 5))
        return sum(check) == 1


if __name__ == '__main__':
    root = tk.Tk()
    app = Gomoku(master=root)
    root.mainloop()
