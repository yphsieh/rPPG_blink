""" 
--------Commands--------
Real-time:      python get_pulse_video.py
                python get_pulse_video.py --subject[name]
Video:          python get_pulse_video.py --video myvid.mp4
------------------------
"""
from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import cv2
import argparse
import numpy as np
import datetime
import time
# TODO: work on serial port comms, if anyone asks for it
# from serial import Serial
import socket
import sys
import pandas as pd

class getPulseApp(object):
    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, args):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)
        serial = args.serial
        baud = args.baud
        video = args.video
        self.kill = False
        self.vidname = ""
        self.send_serial = False
        self.send_udp = False
        if serial:
            self.send_serial = True
            if not baud:
                baud = 9600
            else:
                baud = int(baud)
            self.serial = Serial(port=serial, baudrate=baud)

        udp = args.udp
        if udp:
            self.send_udp = True
            if ":" not in udp:
                ip = udp
                port = 5005
            else:
                ip, port = udp.split(":")
                port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET,  # Internet
                                      socket.SOCK_DGRAM)  # UDP
        if video:
            self.vidname = video

        self.cameras = []
        self.selected_cam = 0

        if args.url is not None:
            camera = Camera(camera=args.url)
            self.cameras.append(camera)

        for i in range(3):
            camera = Camera(camera=i, vid=self.vidname)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break

        self.w, self.h = 0, 0
        self.record = False
        self.sz = (int(self.cameras[self.selected_cam].cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(self.cameras[self.selected_cam].cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.fps = 25
        self.q = 0
        self.out = None
        self.pressed = 0
        # Containerized analysis of recieved image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        # (A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "g": self.start_record,
                             "f": self.stop_record
                             }

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def start_record(self):
        self.processor.start_record = True
        self.processor.bpms = []
        self.processor.ttimes = []
        self.processor.t1 = time.time()
        self.record = True
        self.out = cv2.VideoWriter(args.subject + '_' + str(self.q) + '.mp4', self.fourcc, self.fps, self.sz)
        self.q += 1

    def stop_record(self):
        """
        Writes current data to a csv file
        """
        # fn = str(datetime.datetime.now())
        # fn = fn.replace(":", "_").replace(".", "_")
        fn = args.subject + '_' + str(self.q - 1)
        data = np.vstack((self.processor.ttimes, self.processor.bpms)).T

        df = pd.DataFrame(data=data, columns=['Time', 'BPM'])
        df.to_csv(fn + ".csv")
        
        print("Writing csv")

        self.processor.start_record = False
        if self.record == True:
            self.record = False
            self.out.release()
            print("Saving video: " + fn + '.mp4')

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        # state = self.processor.find_faces.toggle()
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.ttimes,
                 self.processor.bpms],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            if self.record:
                self.stop_record()
                self.record = False
            for cam in self.cameras:
                cam.cam.release()
            if self.send_serial:
                self.serial.close()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        if frame is None:
            self.kill = True
            return

        self.h, self.w, _c = frame.shape

        if self.record:
            self.out.write(frame)

        # else: self.out.release()

        # display unaltered frame
        # imshow("Original",frame)

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        try:
            self.processor.run(self.selected_cam)
        except:
            pass
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        imshow("Processed", output_frame)

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        if self.send_serial:
            self.serial.write(str(self.processor.bpm) + "\r\n")

        if self.send_udp:
            self.sock.sendto(str(self.processor.bpm), self.udp)

        # handle any key presses
        self.key_handler()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('--serial', default=None,
                        help='serial port destination for bpm data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='udp address:port destination for bpm data')
    parser.add_argument('--subject', default='yph')
    parser.add_argument('--video', default="",
                        help='video name')

    parser.add_argument('--url', default=None, type=str,
                        help='IP Webcam url (ex: http://192.168.0.101:8080/video)')

    args = parser.parse_args()

    print(args)

    App = getPulseApp(args)

    while App.kill==False:
        App.main_loop()

    if App.record:
        App.record = False
        App.stop_record()
