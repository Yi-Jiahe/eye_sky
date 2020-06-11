import easytello

from tello_scripts.tello_UI import TelloUI


def main():
    drone = easytello.tello.Tello(debug=False)
    ui = TelloUI(drone)

    # start the Tkinter mainloop
    ui.root.mainloop()

if __name__ == "__main__":
    main()