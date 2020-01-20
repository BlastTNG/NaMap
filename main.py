import src.gui as gui

from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    ex = gui.AppWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()