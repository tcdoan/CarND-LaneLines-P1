import argparse
import cv2 as cv
import numpy as np
import os

# hough.py is the tool to find lane lines from a given image.
# To run the program, just type:
# python hough.py -i <image_file_name>
class Hough:
    def __init__(self, img, polygon_pts):
        self.img = img

        self.polygon_pts = polygon_pts
        self.polygon_capture = []

        self.blur_ksize = 5
        self.blur_sigmaX = 1
        self.blur_sigmaY = 0
        self.canny_threshold_low = 50
        self.canny_threshold_hi = 150

        # Distance and angular resolution of grid in Hough space.
        self.rho = 2
        self.theta = np.pi/180

        # Minimum number of votes a candidate line needs to have
        self.threshold = 15

        # Minimum length of a line (in pixels) accepted
        self.min_line_length = 40

        # Maximum distance between segments allowed to be connected into a single line
        self.max_line_gap = 20

        def onchangeBlurKernelSize(value):
            self.blur_ksize = value
            # Make sure it is an odd number
            self.blur_ksize += (self.blur_ksize + 1) %2
            self.line_segments()

        def onchangeBlurSigmaX(value):
            self.blur_sigmaX = value
            self.line_segments()

        def onchangeBlurSigmaY(value):
            self.blur_sigmaY = value
            self.line_segments()

        def onchangeCannyThresHoldLow(value):
            self.canny_threshold_low = value
            self.line_segments()

        def onchangeCannyThresHoldHi(value):
            self.canny_threshold_hi = value
            self.line_segments()

        def onchangeRho(value):
            self.rho = value
            self.line_segments()

        def onchangeThreshold(value):
            self.threshold = value
            self.line_segments()

        def onchangeMinLineLen(value):
            self.min_line_length = value
            self.line_segments()

        def onchangeMaxLineGap(value):
            self.max_line_gap = value
            self.line_segments()

        def click_for_roi(event, x, y, flags, param):
            if event == cv.EVENT_RBUTTONDOWN:
                self.polygon_capture = []

            if event == cv.EVENT_LBUTTONDBLCLK:
                self.polygon_capture.append((x, y))
                if len(self.polygon_capture) > 3:
                    self.polygon_pts = self.polygon_capture
                    print('Polygon_capture:  ' + str(self.polygon_capture))
                    self.line_segments()

        cv.namedWindow('edges', cv.WINDOW_NORMAL)
        cv.namedWindow('hough', cv.WINDOW_NORMAL)

        cv.setMouseCallback('edges', click_for_roi)
        cv.createTrackbar("blur_ksize", 'edges', self.blur_ksize, 19, onchangeBlurKernelSize)
        cv.createTrackbar("blur_sigmaX", 'edges', self.blur_sigmaX, 20, onchangeBlurSigmaX)
        cv.createTrackbar("blur_sigmaY", 'edges', self.blur_sigmaY, 20, onchangeBlurSigmaY)
        cv.createTrackbar("canny_threshold_low", 'edges', self.canny_threshold_low, 255, onchangeCannyThresHoldLow)
        cv.createTrackbar("canny_threshold_hi", 'edges', self.canny_threshold_hi, 255, onchangeCannyThresHoldHi)
        
        cv.createTrackbar("rho", 'hough', self.rho, 100, onchangeRho)
        cv.createTrackbar("threshold", 'hough', self.threshold, 100, onchangeThreshold)
        cv.createTrackbar("min_line_length", 'hough', self.min_line_length, 100, onchangeMinLineLen)
        cv.createTrackbar("max_line_gap", 'hough', self.max_line_gap, 100, onchangeMaxLineGap)

        self.line_segments()
        print("Change the parameters from trackbar UI to fine tune.  Press a key to close.")
        cv.waitKey(0)
        cv.destroyAllWindows()

    def line_segments(self):
        blur_gray = cv.GaussianBlur(self.img, (self.blur_ksize, self.blur_ksize), self.blur_sigmaX, self.blur_sigmaY)

        self.edge_img = cv.Canny(blur_gray, self.canny_threshold_low, self.canny_threshold_hi)

        mask  = np.zeros_like(self.edge_img)
        vertices = np.array([self.polygon_pts], dtype=np.int32)
        cv.fillPoly(mask, vertices, 255)
        masked_edges = cv.bitwise_and(self.edge_img, mask)

        self.lines = cv.HoughLinesP(masked_edges, self.rho, self.theta, self.threshold,
                                np.array([]), self.min_line_length, self.max_line_gap)

        line_image = np.copy(self.img)*0
        line_image = np.dstack((line_image, line_image, line_image))

        print("Num lines:", len(self.lines))
        for line in self.lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # create a "color" binary image to combine with line image.
        color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
        self.combo = cv.addWeighted(color_edges, 0.8, line_image, 1, 0)
        cv.imshow('edges', self.edge_img)
        cv.imshow('hough', self.combo)

# Return
# - Number of line segments
# - Distance of 1 shortest segment
# - Distance of 2 longest  segments
# - Ratio of sum(2 longest segments) over number of line segments
def lines_stats(lines):
    data = [] 
    for line in lines:
        for x1, y1, x2, y2 in line:
            dist = round(np.linalg.norm(np.array([x1,y1]) - np.array([x2, y2])))
            data.append((x1, y1, x2, y2, dist))   
    data.sort(key=lambda tuple: tuple[4])
    numSegments =  len(data)
    shortest = data[0]    
    secondLongest = data[numSegments-2]
    longest = data[numSegments-1]
    ratio = (longest[4] +  secondLongest[4]) // numSegments
    return numSegments, shortest[4], secondLongest[4], longest[4], ratio

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to the image")
    args = vars(parser.parse_args())

    img = cv.imread(args["image"], cv.IMREAD_GRAYSCALE)
    poly_corners = [(0, img.shape[0]), (450, 290), (490, 290), (img.shape[1], img.shape[0])]

    cv.imshow('original', img)
    hough = Hough(img, poly_corners)

    print("Fine tuned parameters:")
    print("- image: ", args["image"])
    print("- polygon_pts:", hough.polygon_pts)
    print("- blur_ksize:", hough.blur_ksize)
    print("- blur_sigmaX:", hough.blur_sigmaX)
    print("- blur_sigmaY:", hough.blur_sigmaY)
    print("- canny_threshold_low:", hough.canny_threshold_low)
    print("- canny_threshold_hi:", hough.canny_threshold_hi)
    print("- rho:", hough.rho)
    print("- threshold:", hough.threshold)
    print("- min_line_length:", hough.min_line_length)
    print("- max_line_gap:", hough.max_line_gap)
    print("- lines_stats:", lines_stats(hough.lines))

    path, fname = os.path.split(args["image"])
    file, ext = os.path.splitext(fname)
    edges_filename = os.path.join("output", file + "-edges" + ext)
    lines_filename = os.path.join("output", file + "-lines" + ext) 
    cv.imwrite(edges_filename, hough.edge_img)
    cv.imwrite(lines_filename, hough.combo)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
