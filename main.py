import src.gui as gui

from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
<<<<<<< HEAD
<<<<<<< HEAD
    ex = gui.MainWindow()
=======
    ex = gui.App()
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
    ex = gui.App()
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()