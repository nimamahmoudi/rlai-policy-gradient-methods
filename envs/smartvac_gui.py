from PyQt5 import QtWidgets, QtGui

from .environment import SmartVac

import sys
import threading
import time

__version__ = '0.1'


def create_application():
    global app
    app = QtWidgets.QApplication(sys.argv)


def run_application():
    app.exec_()


def close_application():
    app.quit()


class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.cwidth = 160
        self.cheight = 160

        self.title = 'Smart Vaccuum'
        self.setWindowTitle(self.title)
        self.setFixedSize(800, 321)

        self.bg = QtWidgets.QLabel(self)
        self.bg.setGeometry(0, 0, 800, 321)
        self.bg.setText("")
        self.bg.setObjectName("background")
        self.bg.setPixmap(QtGui.QPixmap('bg.png'))

        self.agent = QtWidgets.QLabel(self)
        self.agent.setGeometry(0, 0, 156, 156)
        self.agent.setText("")
        self.agent.setPixmap(QtGui.QPixmap('agent.png'))

    def set_agent_position(self, i, j):
        self.agent.setGeometry(i * self.cwidth, j * self.cheight, 156, 156)


import numpy as np

import io
import cv2

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QBuffer


def get_screen_np_image():
    screen = app.primaryScreen()
    p = screen.grabWindow(ex.winId())
    image = p.toImage()

    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "png")
    img_stream = io.BytesIO(buffer.data())
    img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)

    return img


class Consumer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        time.sleep(0.1)
        import numpy as np
        import random

        ##########
        from datetime import datetime
        date = datetime.now()
        filename = date.strftime('%Y-%m-%d_%H-%M-%S.mp4')

        img = get_screen_np_image()
        fps = 10
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (img.shape[1], img.shape[0]))

        sv = SmartVac()
        episode_count = 10
        episode_rewards = np.zeros(episode_count)

        for i_episode in range(episode_count):
            done = False
            totalReward = 0

            obs = sv.reset()

            ex.set_agent_position(sv.x, sv.y)
            time.sleep(.1)

            while not done:
                obs, reward, done = sv.step(random.choice(list(range(4))))

                ex.set_agent_position(sv.x, sv.y)
                # time.sleep(.1)

                out.write(get_screen_np_image())

                if done:
                    time.sleep(.1)
                    for _ in range(int(1 * fps)):
                        out.write(get_screen_np_image())

                totalReward += reward

            episode_rewards[i_episode] = totalReward
            print('Reward: ', totalReward)

        out.release()

        print('Done!')
        close_application()


if __name__ == '__main__':
    create_application()

    ex = App()
    ex.show()

    consumer = Consumer()
    consumer.start()

    run_application()
    consumer.join()
