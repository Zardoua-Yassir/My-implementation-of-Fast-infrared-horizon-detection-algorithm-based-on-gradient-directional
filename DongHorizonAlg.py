"""
Class implementing the algorithm described in paper https://doi.org/10.1364/JOSAA.402620
Implementation by Yassir Zardoua. Email: yassirzardoua@gmail.com
"""
import cv2 as cv
import numpy as np
import os
from time import time
from warnings import warn
from math import pi, atan, sin, cos, isinf
import scipy.signal as signal
import scipy.ndimage as ndimage
from dictances import bhattacharyya
from matplotlib import pyplot as plt
import sklearn


class DongL:
    def __init__(self, SDGD_th=18):
        # constants
        self.samples_nbr = 1000000000
        self.SDGD_th = SDGD_th
        self.processed_image_width = 640
        self.processed_image_height = 512
        self.color_red = (0, 0, 255)
        self.color_blue = (255, 0, 0)
        self.color_green = (0, 255, 0)
        self.color_yellow = (0, 255, 255)
        self.color_aqua = (255, 255, 0)
        self.colorlist = [self.color_red, self.color_blue, self.color_aqua, self.color_green, self.color_yellow]

        # Algorithm parameters
        self.l = 50  # the width of windows W1, W2, ...
        self.W_Ys = np.int32(np.arange(0, 475, self.l / 2))  # starting index of all windows W1, W2, ...
        self.W_Ye = np.int32(np.arange(self.l, 525, self.l / 2))  # ending index of all windows W1, W2, ...
        self.Q_Ys = np.int32([])  # starting index of all potential sea-sky regions Q1, Q2, ...
        self.Q_Ye = np.int32([])  # ending index of all windows Q1, Q2, ...
        self.B = np.float32([])  # a vector containing the Bhattacharyyaa coefficients of all potential sea-sky regions
        # Qp. Thus, its shape is equal to that of self.Q_Ys

        self.Rn = np.zeros(shape=(self.W_Ys.shape))  # a vector containing the maximum value of self.MVR - the minumum
        # value of self.MVR inside the k-th window Wk. See Rn in the paper
        self.K = 0.3 * 255  # threshold for selecting potential sea-sky regions

        # Parameters related to detection of line segments
        self.Gx_kernel = np.array([[-1, 1],
                                   [-1, 1]])

        self.Gy_kernel = np.array([[-1, -1],
                                   [1, 1]])

        # image results
        self.img_input = None
        self.res_img = None
        self.img_with_roh = None  # original image with region of horizon boundary
        self.img_with_horizon = None
        self.roh = None

        self.res_width = 640
        self.res_height = 512
        self.org_width = 1920
        self.org_height = 1080

        # RANSAC attributes
        self.M = 0  # the total number of reserved/survived segments
        self.minimum_error = 10000000000000  # minimum updated ransac error
        self.ransac_sample_index = 0  # index to a random reserved segment endpoints (starting and ending point)
        self.x_inlier = np.array([], dtype=np.int32)  # coordinates of points whose distance to the current line is
        # smaller than a custom threshold (threshold distance is expressed in equation 12)
        self.y_inlier = np.array([], dtype=np.int32)

        self.ransac_score = 0
        self.k = None  # slope of current ransac line; will store the final line
        self.b = None  # intercept of current ransac line; will store the final line

        self.hl_slope_org = None  # slope of the final horizon line on the original image
        self.hl_intercept_org = None  # intercept of the final horizon line on the original image

        self.hl_slope_res = None  # slope of the final horizon line on the processed/resized image
        self.hl_intercept_res = None  # intercept of the final horizon line on the processed/resized image
        self.Y = None  # Position on the original image size
        self.phi = None  # tilt on the original image size

        self.survived_segs_midpoint_x = np.array([], dtype=np.int32)
        self.survived_segs_midpoint_y = np.array([], dtype=np.int32)

        self.survived_segs_k = np.array([], dtype=np.int32)  # slope of each survived segment (as a vector)
        self.survived_segs_b = np.array([], dtype=np.int32)  # intercept of each survived segment (as a vector)
        self.length_weighted_error = None  # ransac error of the current candidate line weighted by segments lengths
        # computed by setting p = 0.99 (success prob), s = 2 (number of samples), e = 0.5 (outliers ratio)

    def get_horizon(self, img):
        """
        :param img: source image
        :param get_image: if True, the horizon is drawn on the attribute 'self.img_with_hl'
        :return:
        """
        self.img_rgb_org = img
        self.img_with_roh = cv.resize(img, (640, 512))
        self.img_with_segs = cv.resize(img, (640, 512))
        self.img_with_filtered_segs = cv.resize(img, (640, 512))

        self.start_time = time()  # to measure the computational time
        self.img_input = img
        if len(self.img_input.shape) == 3:
            self.img_input_gray = cv.cvtColor(self.img_input, cv.COLOR_BGR2GRAY)
        elif len(self.img_input.shape) == 2:
            self.img_input_gray = self.img_input
        else:
            raise ValueError("Invalid shape of argument img. img must be an RGB/BGR or Grayscale image")
        self.get_roh()  # This method stores the ROI in self.roh
        self.extract_segments()
        self.stretch_segments()
        self.filter_segments()  # using the gradient direction filtering method
        self.fit_horizon()
        self.convert_hl_pars()
        self.latency = round(time() - self.start_time, 4)
        print("time execution is: ", self.latency, " seconds")
        return self.Y, self.phi, self.latency, self.detected_hl_flag

    def get_roh(self):
        """
        Extracts the ROH (Region Of Horizon) from self.img_input_gray, as described in section B. Rough Extraction of
        Sea–Sky Region of the paper
        :return: an image overlaid by ROH
        """
        self.compute_MVR()
        self.compute_Rn()
        self.get_roh_xy()
        self.roh = self.img_input_gray_resized[self.roh_ys:self.roh_ye, :]

        # Find indexes where Rn is higher than K = 0.3 * 255.

        # Get all potential sea-sky regions (Q1, Q2, ...) using windows Wk whose Rn is higher than threshold K
        # For each region Q, compute the Bhattacharyya distance (see equation 2 in the paper)

        # Extract ROH as the potential sea-sky region whose Bhattacharyya distance is the smallest (yes the smallest)
        return self.img_with_roh

    def compute_MVR(self):
        """
        Computes the Mean Value of Rows of an image, then applies a median smoothing. The result is stored in self.MVR
        """
        # resize the input image to size 640x512
        self.img_input_gray_resized = cv.resize(self.img_input_gray, (640, 512))

        # compute for each row the arithmetic mean + apply median smoothing
        self.MVR = np.mean(self.img_input_gray_resized, axis=1)  # computes the mean value of pixel intensities of each
        self.MVR = signal.medfilt(self.MVR, kernel_size=5)  # the kernel size is not given in the original paper

    def compute_Rn(self):
        """
        Computes Rn = max(windowed_MVR) - min(windowed_MVR). The result is stored in self.Rn
        """
        # iterate over slices of self.MVR. Each slice represents elements windowed by W1, W2, ..., Wn (see Fig.3.).
        # The width of each window Wk is l = 50 pixels and the stride between them is l/2 = 25 pixels
        idx = 0
        for ys, ye in zip(self.W_Ys, self.W_Ye):
            # During each iteration, compute vector Rn = (max(windowed_mvr) - min(windowed_mvr)).
            # Rn.shape = (Number of windows Wk,)
            self.windowed_MVR = self.MVR[ys:ye]
            self.Rn[idx] = np.max(self.windowed_MVR) - np.min(self.windowed_MVR)
            idx = idx + 1

    def get_roh_xy(self):
        """
        Finds the xy coordinates of the region of horizon.
        Steps are:
            1) find potential sea-sky regions (Q1, Q2,...Qp,...) as the region whose Rn>K. Authors didn't explain how to
            handle the case where no region Qp satisfies Rn > K. In such a case, we take the roh as the region with the
            highest Rn.  Also, if only one region Qp is detected, we directly consider it as a the roh because it is the
            only candidate.
            2) divide each region Qp into two equal regions Mp1 and Mp2, (vertically distributed)
            3) compute the Bhathacharyya distance between each pair Mp1, Mp2. The region Qp with the smallest distance
            is the region of horizon.
        """
        self.Q_indexes = np.where(self.Rn > self.K)[0]  # Indexes of candidate regions Qp. These indexes should be used
        # to index self.W_Ys and self.W_Ye
        self.number_of_Qs = self.Q_indexes.shape[0]  # number of potential sea sky regions

        self.Q_Ys = self.W_Ys[self.Q_indexes]
        self.Q_Ye = self.W_Ye[self.Q_indexes]

        if self.number_of_Qs == 0:  # True if no candidate region Q is selected
            # in that case, get the region with the maximal value of Rn
            self.Q_roh_index = np.argmax(self.Rn)
            self.roh_ys = self.W_Ys[self.Q_roh_index]
            self.roh_ye = self.W_Ye[self.Q_roh_index]
        elif self.number_of_Qs == 1:  # True if only one candidate region Q is selected.
            self.Q_roh_index = self.Q_indexes
            self.roh_ys = self.W_Ys[self.Q_roh_index][0]
            self.roh_ye = self.W_Ye[self.Q_roh_index][0]
        elif self.number_of_Qs > 1:  # True if more than one candidate region Q is selected
            # Compute the Bhattacharyya coefficient (see equation 2)
            # self.Q_Ys = self.W_Ys[self.Q_indexes]
            # self.Q_Ye = self.W_Ye[self.Q_indexes]
            self.B = np.zeros(shape=self.Q_Ys.shape, dtype=np.float32)  #

            idx = 0
            for ys, ye in zip(self.Q_Ys, self.Q_Ye):
                self.ys_Mi1 = ys
                self.ye_Mi1 = ys + int((ye - ys) / 2)

                self.ys_Mi2 = self.ye_Mi1
                self.ye_Mi2 = ye

                self.Mi1 = self.img_input_gray_resized[self.ys_Mi1:self.ye_Mi1, :]
                self.Mi2 = self.img_input_gray_resized[self.ys_Mi2:self.ye_Mi2, :]

                self.Pi1, _ = np.histogram(a=self.Mi1, bins=255)
                self.Pi2, _ = np.histogram(a=self.Mi2, bins=255)

                # normalizing histograms
                self.Pi1 = np.divide(self.Pi1, 16000)  # 16000 is the total number of  pixels in image block Mi1
                self.Pi2 = np.divide(self.Pi2, 16000)  # 16000 is the total number of  pixels in image block Mi2

                self.B[idx] = np.dot(np.sqrt(self.Pi1), np.sqrt(self.Pi2))  # this is equivalent to equation 2
                idx = idx + 1

                # cv.imwrite("Cropped.png", self.img_input_gray_resized[ys:ye, :])
                # cv.imwrite("Cropped_Mi1.png", self.Mi1)
                # cv.imwrite("Cropped_Mi2.png", self.Mi2)
            self.Q_roh_index = np.argmin(self.B)
            self.roh_ys = self.Q_Ys[self.Q_roh_index]
            self.roh_ye = self.Q_Ye[self.Q_roh_index]

        self.roh_xs = int(0)
        self.roh_xe = int(639)  # 640 - 1

    def get_roh_test(self, src, dst, thickness=1):
        """
        You can use this API to detect and draw regions of horizons.
        :param src: folder of image samples
        :param dst: where to store image samples overlaid by the extracted region of horizon (roh)
        :param thickness: thickness of the lines used to overlay the roh
        :return:
        """

        src_files = os.listdir(src)
        for src_file in src_files:
            img = cv.imread(os.path.join(src, src_file))
            self.start_time = time()  # to measure the computational time
            self.img_with_roh = cv.resize(img, (640, 512))
            self.img_input = img
            if len(self.img_input.shape) == 3:
                self.img_input_gray = cv.cvtColor(self.img_input, cv.COLOR_BGR2GRAY)
            elif len(self.img_input.shape) == 2:
                self.img_input_gray = self.img_input
            else:
                raise ValueError("Invalid shape of argument img. img must be an RGB/BGR or Grayscale image")

            self.compute_MVR()
            self.compute_Rn()
            self.get_roh_xy()
            self.execution_time = round(time() - self.start_time, 4)

            # Draw and save
            cv.line(img=self.img_with_roh, pt1=(self.roh_xs, self.roh_ys), pt2=(self.roh_xe, self.roh_ys),
                    color=(0, 0, 255), thickness=thickness)
            cv.line(img=self.img_with_roh, pt1=(self.roh_xs, self.roh_ye), pt2=(self.roh_xe, self.roh_ye),
                    color=(0, 0, 255), thickness=thickness)
            self.img_with_roh_org_size = cv.resize(src=self.img_with_roh, dsize=(1920, 1080))
            cv.imwrite(os.path.join(dst, "ROH_" + src_file), self.img_with_roh_org_size)
        return self.img_with_roh

    def extract_segments(self):
        """
        This method detects line segments in the following way:
        Step 1: compute the gradient of the roh
        Step 2: divide the roh and its gradient magnitude (must have 50 rows) into 5 sub-images of 10 rows each
        Step 3:
              a- compute adaptive threshold of the i-th sub-image
              b- extract line segments
              c- identify segments' endpoints on the coordinate system of image self.roh
        :return: extracted segments as self.roh_segs
        """
        # Step 1
        self.Gx = ndimage.convolve(np.float32(self.roh), self.Gx_kernel)  # self.roh must be floated, otherwise, self.Gx
        # gets clipped into [0, 255].
        self.Gy = ndimage.convolve(np.float32(self.roh), self.Gy_kernel)
        self.roh_grad_mag = np.sqrt(np.square(self.Gx) + np.square(self.Gy))

        # Step 2: extract sub-images from roh and its gradient magnitude
        self.roh_sub_1 = self.roh[00:10, :]
        self.roh_sub_2 = self.roh[10:20, :]
        self.roh_sub_3 = self.roh[20:30, :]
        self.roh_sub_4 = self.roh[30:40, :]
        self.roh_sub_5 = self.roh[40:50, :]

        self.roh_sub_list = [self.roh_sub_1, self.roh_sub_2, self.roh_sub_3, self.roh_sub_4, self.roh_sub_5]

        self.roh_grad_mag_sub_1 = self.roh_grad_mag[00:10, :]
        self.roh_grad_mag_sub_2 = self.roh_grad_mag[10:20, :]
        self.roh_grad_mag_sub_3 = self.roh_grad_mag[20:30, :]
        self.roh_grad_mag_sub_4 = self.roh_grad_mag[30:40, :]
        self.roh_grad_mag_sub_5 = self.roh_grad_mag[40:50, :]

        self.roh_grad_mag_list = [self.roh_grad_mag_sub_1, self.roh_grad_mag_sub_2, self.roh_grad_mag_sub_3,
                                  self.roh_grad_mag_sub_4, self.roh_grad_mag_sub_5]

        # Step 3-a
        self.roh_segs = np.zeros((0, 1, 4), dtype=np.float32)
        self.y_offset = self.roh_ys  # offset to get segment endpoints on the coordinate system of the original
        # resolution 512x640
        for y_offset, self.roh_sub_i, self.roh_grad_mag_i in zip(range(self.roh_ys, self.roh_ye, 10), self.roh_sub_list,
                                                                 self.roh_grad_mag_list):
            self.grad_th = np.divide(np.mean(self.roh_grad_mag_i), np.max(self.roh_grad_mag_i))
            # self.grad_th = self.grad_th * 255
            self.fsd = cv.ximgproc.createFastLineDetector(_length_threshold=1,
                                                          _canny_th1=self.grad_th,
                                                          _canny_th2=self.grad_th)

            self.roh_sub_i_segs = self.fsd.detect(self.roh_sub_i)  # (x1, y1, x2, y2)
            if self.roh_sub_i_segs is None:
                continue
            self.roh_sub_i_segs[:, :, 1] = np.add(self.roh_sub_i_segs[:, :, 1], y_offset)  # offset ys coordinates
            self.roh_sub_i_segs[:, :, 3] = np.add(self.roh_sub_i_segs[:, :, 3], y_offset)  # offset ye coordinates
            self.roh_segs = np.concatenate((self.roh_segs, self.roh_sub_i_segs), axis=0)

        self.total_segments = self.roh_segs.shape[0]  # get the number of all detected segments: self.N_a
        self.roh_segs = np.reshape(self.roh_segs, newshape=(self.total_segments, 4))  # reshape from
        # (self.total_segments, 1, 4) to (self.total_segments, 4). This is easier for consequent processing.

    def stretch_segments(self):
        """
        stretches extracted segments by a factor of k = 0.5.
        :return:
        """
        self.roh_segs_xs, self.roh_segs_ys = self.roh_segs[:, 0], self.roh_segs[:, 1]
        self.roh_segs_xe, self.roh_segs_ye = self.roh_segs[:, 2], self.roh_segs[:, 3]

        # length of extracted segments
        self.roh_segs_len = np.sqrt(np.add(np.square(np.subtract(self.roh_segs_xs, self.roh_segs_xe)),
                                           np.square(np.subtract(self.roh_segs_ys, self.roh_segs_ye))))

        self.roh_segs_tilt_rad = np.divide(np.subtract(self.roh_segs_ye, self.roh_segs_ys),
                                           np.subtract(self.roh_segs_xe, self.roh_segs_xs))
        self.roh_segs_tilt_rad = np.arctan(self.roh_segs_tilt_rad)

        self.DX = np.abs(np.cos(self.roh_segs_tilt_rad) * 0.5 * self.roh_segs_len)
        self.DY = np.abs(np.sin(self.roh_segs_tilt_rad) * 0.5 * self.roh_segs_len)

        # Add the offset DX on all x coordinates (Since addition must be done only on rightmost x endpoint of a segment,
        # we add DX on all endpoints x, then we correct the leftmost x endpoints by subtracting DX from them later.
        self.roh_streched_segs_xs = self.roh_segs_xs + self.DX
        idxs_to_correct = np.where(self.roh_segs_xs < self.roh_segs_xe)[0]
        self.roh_streched_segs_xs[idxs_to_correct] = self.roh_segs_xs[idxs_to_correct] - self.DX[idxs_to_correct]

        self.roh_streched_segs_xe = self.roh_segs_xe + self.DX
        idxs_to_correct = np.where(self.roh_segs_xe < self.roh_segs_xs)[0]
        self.roh_streched_segs_xe[idxs_to_correct] = self.roh_segs_xe[idxs_to_correct] - self.DX[idxs_to_correct]

        # Add the offset DY on all y coordinates (Since addition must be done only on lowermost y endpoint of a segment,
        # we add DY on all endpoints y, then we correct the uppermost y endpoints by subtracting DY from them later.
        self.roh_streched_segs_ys = self.roh_segs_ys + self.DY
        idxs_to_correct = np.where(self.roh_segs_ys < self.roh_segs_ye)[0]
        self.roh_streched_segs_ys[idxs_to_correct] = self.roh_segs_ys[idxs_to_correct] - self.DY[idxs_to_correct]

        self.roh_streched_segs_ye = self.roh_segs_ye + self.DY
        idxs_to_correct = np.where(self.roh_segs_ye < self.roh_segs_ys)[0]
        self.roh_streched_segs_ye[idxs_to_correct] = self.roh_segs_ye[idxs_to_correct] - self.DY[idxs_to_correct]

        self.roh_streched_segs = np.zeros(shape=self.roh_segs.shape)

        self.clip_stretched_segs()
        self.roh_streched_segs[:, 0] = self.roh_streched_segs_xs
        self.roh_streched_segs[:, 1] = self.roh_streched_segs_ys
        self.roh_streched_segs[:, 2] = self.roh_streched_segs_xe
        self.roh_streched_segs[:, 3] = self.roh_streched_segs_ye

        # todo establish vector DX and DY

        # todo strech segments by computing new coordinates using DX and DY

        # todo exclude segments stretching outside the image boundaries

    def clip_stretched_segs(self):
        """
        This method clips stretched segments extending outside the image box by replacing them with the corresponding
        non-streched coordinates
        :return:
        """
        # verifying the starting point ps = (xs, ys)
        ps_clip_condition_1 = self.roh_streched_segs_xs > (self.processed_image_width - 1)
        ps_clip_condition_2 = self.roh_streched_segs_xs < 0
        ps_clip_condition_3 = self.roh_streched_segs_ys > (self.processed_image_height - 1)
        ps_clip_condition_4 = self.roh_streched_segs_ys < 0

        indxs_to_clip = np.concatenate((np.where(ps_clip_condition_1)[0],
                                        np.where(ps_clip_condition_2)[0],
                                        np.where(ps_clip_condition_3)[0],
                                        np.where(ps_clip_condition_4)[0]), axis=0)
        if len(indxs_to_clip) > 0:
            self.roh_streched_segs_xs[indxs_to_clip] = self.roh_segs_xs[indxs_to_clip]
            self.roh_streched_segs_ys[indxs_to_clip] = self.roh_segs_ys[indxs_to_clip]

        # verifying the ending point pe = (xe, ye)
        pe_clip_condition_1 = self.roh_streched_segs_xe > (self.processed_image_width - 1)
        pe_clip_condition_2 = self.roh_streched_segs_xe < 0
        pe_clip_condition_3 = self.roh_streched_segs_ye > (self.processed_image_height - 1)
        pe_clip_condition_4 = self.roh_streched_segs_ye < 0

        indxs_to_clip = np.concatenate((np.where(pe_clip_condition_1)[0],
                                        np.where(pe_clip_condition_2)[0],
                                        np.where(pe_clip_condition_3)[0],
                                        np.where(pe_clip_condition_4)[0]))
        if len(indxs_to_clip) > 0:
            self.roh_streched_segs_xe[indxs_to_clip] = self.roh_segs_xe[indxs_to_clip]
            self.roh_streched_segs_ye[indxs_to_clip] = self.roh_segs_ye[indxs_to_clip]

    def draw_segments(self, img_dst, seg_set, colorlist, thickness=1, save=False):
        """
        Description:
        ------------
        Draws line segments given in 'segments' on 'img_dst'.

        Parameters:
        -----------
        :param dst_folder: folder where to save the result
        :param filename: filename
        :param img_dst: image on which to draw segments
        :param segments: numpy arrays of shape (N,4).
        :param colors: a list containing color tuples
        :param thickness: thickness of segments to draw
        Usage Example:
        --------------
            segments = [self.Segs_a, self.Segs_b]
            colors = [(0, 0, 255), (0, 255, 0)]
            self.img_segs = self.draw_segs(img_dst=self.in_img_bgr, segments=segments, colors=colors)
            cv.imwrite("result.png", self.img_segs)

        :return: img_dst with drawn segments
        """
        img_dst = np.float32(img_dst)
        nbr_of_colors = len(colorlist) - 1
        colorindex = 0
        for points in seg_set:
            xs = int(points[0])
            ys = int(points[1])
            xe = int(points[2])
            ye = int(points[3])
            cv.line(img_dst, (xs, ys), (xe, ye), color=colorlist[colorindex],
                    thickness=thickness)
            if colorindex < nbr_of_colors:
                colorindex += 1
            else:
                colorindex = 0
        imgSegments = np.uint8(img_dst)
        if save:
            cv.imwrite("imgSegments.png", np.uint8(img_dst))
            # cv.imwrite("cannyth1_" + str(self.canny_th1) + "_imgSegments.png", np.uint8(img_dst))
        return imgSegments

    def test_extracted_segments(self, src, dst, thickness=2):
        src_files = os.listdir(src)
        index = 0
        for src_file in src_files:
            img = cv.imread(os.path.join(src, src_file))
            self.start_time = time()  # to measure the computational time
            self.img_with_roh = cv.resize(img, (640, 512))
            self.img_with_segs = cv.resize(img, (640, 512))
            self.img_with_filtered_segs = cv.resize(img, (640, 512))
            self.img_input = img
            if len(self.img_input.shape) == 3:
                self.img_input_gray = cv.cvtColor(self.img_input, cv.COLOR_BGR2GRAY)
            elif len(self.img_input.shape) == 2:
                self.img_input_gray = self.img_input
            else:
                raise ValueError("Invalid shape of argument img. img must be an RGB/BGR or Grayscale image")

            self.compute_MVR()
            self.compute_Rn()
            self.get_roh_xy()
            self.get_roh()  # This method stores the ROI in self.roh
            self.extract_segments()
            colorlist = [self.color_red, self.color_blue, self.color_aqua, self.color_green, self.color_yellow]
            self.img_with_segs = self.draw_segments(img_dst=self.img_with_segs, seg_set=self.roh_segs,
                                                    colorlist=colorlist)

            # Draw and save
            cv.line(img=self.img_with_segs, pt1=(self.roh_xs, self.roh_ys), pt2=(self.roh_xe, self.roh_ys),
                    color=(0, 0, 255), thickness=thickness)
            cv.line(img=self.img_with_segs, pt1=(self.roh_xs, self.roh_ye), pt2=(self.roh_xe, self.roh_ye),
                    color=(0, 0, 255), thickness=thickness)
            self.img_with_segs_org_size = cv.resize(src=self.img_with_segs, dsize=(1920, 1080))
            cv.imwrite(os.path.join(dst, "Segs_ROH_" + src_file), self.img_with_segs_org_size)
            # cv.imwrite(os.path.join(dst, "Segs_ROH_" + src_file), self.img_with_segs)
            self.execution_time = round(time() - self.start_time, 4)
            index = index + 1
            if index == self.samples_nbr:
                return
        return 0

    def test_stretched_segments(self, src, dst, thickness=2):
        src_files = os.listdir(src)
        index = 0
        for src_file in src_files:
            img = cv.imread(os.path.join(src, src_file))
            self.start_time = time()  # to measure the computational time
            self.img_with_roh = cv.resize(img, (640, 512))
            self.img_with_segs = cv.resize(img, (640, 512))
            self.img_input = img
            if len(self.img_input.shape) == 3:
                self.img_input_gray = cv.cvtColor(self.img_input, cv.COLOR_BGR2GRAY)
            elif len(self.img_input.shape) == 2:
                self.img_input_gray = self.img_input
            else:
                raise ValueError("Invalid shape of argument img. img must be an RGB/BGR or Grayscale image")

            self.compute_MVR()
            self.compute_Rn()
            self.get_roh_xy()
            self.get_roh()  # This method stores the ROI in self.roh
            self.extract_segments()
            self.stretch_segments()

            colorlist = [self.color_red, self.color_blue, self.color_aqua, self.color_green, self.color_yellow]
            self.img_with_segs = self.draw_segments(img_dst=self.img_with_segs, seg_set=self.roh_streched_segs,
                                                    colorlist=colorlist)

            # Draw and save
            cv.line(img=self.img_with_segs, pt1=(self.roh_xs, self.roh_ys), pt2=(self.roh_xe, self.roh_ys),
                    color=(0, 0, 255), thickness=thickness)
            cv.line(img=self.img_with_segs, pt1=(self.roh_xs, self.roh_ye), pt2=(self.roh_xe, self.roh_ye),
                    color=(0, 0, 255), thickness=thickness)
            self.img_with_segs_org_size = cv.resize(src=self.img_with_segs, dsize=(1920, 1080))
            cv.imwrite(os.path.join(dst, "Streched_Segs_" + src_file), self.img_with_segs_org_size)
            # cv.imwrite(os.path.join(dst, "Segs_ROH_" + src_file), self.img_with_segs)
            self.execution_time = round(time() - self.start_time, 4)
            index = index + 1
            if index == self.samples_nbr:
                return
        return 0

    def filter_segments(self):
        """
        Filters stretched segments based on their SDGD expressed in equation 4. The filter keeps only segments with an
        SDGD < 18.
        :return:
        """
        self.x_out = np.zeros((0,))  # initialize coordinates of output edge pixels
        self.y_out = np.zeros((0,))  # initialize coordinates of output edge pixels

        self.roh_streched_segs_len = np.sqrt(
            np.add(np.square(np.subtract(self.roh_streched_segs_xs, self.roh_streched_segs_xe)),
                   np.square(np.subtract(self.roh_streched_segs_ys, self.roh_streched_segs_ye))))

        self.roh_streched_segs_len = np.uint16(np.subtract(self.roh_streched_segs_len, 1))  # All length values must be
        # integers to allow subsequent processing. Subtracting 1 allows subtraction of 1 for self.roh_streched_segs_len
        # iterations.

        # self.u is a vector whose length cannot be surpassed by
        self.u = np.arange(0, np.sqrt(np.add(np.square(self.processed_image_width),
                                             np.square(self.processed_image_height))))
        # the length of any possible segment.

        # This angle is in degree because the filtering threshold is 18, which can't be obtained unless degree is used
        self.roh_grad_angle = np.multiply(np.arctan(np.abs(np.divide(self.Gy, self.Gx))), 180 / np.pi)

        self.number_of_segments = self.roh_streched_segs.shape[0]
        self.SDGD = np.zeros(shape=(self.number_of_segments,))
        index = 0
        for roh_streched_segs_len_i, xs_i, ys_i, xe_i, ye_i in zip(self.roh_streched_segs_len,
                                                                   self.roh_streched_segs_xs,
                                                                   self.roh_streched_segs_ys,
                                                                   self.roh_streched_segs_xe,
                                                                   self.roh_streched_segs_ye):
            self.u_n = self.u[0:roh_streched_segs_len_i]

            # xy coordinates of all points along a given segment are stored in self.x_i, self.y_i
            self.x_i = np.add(np.multiply(np.divide(np.subtract(xe_i, xs_i), roh_streched_segs_len_i), self.u_n), xs_i)
            self.y_i = np.add(np.multiply(np.divide(np.subtract(ye_i, ys_i), roh_streched_segs_len_i), self.u_n), ys_i)

            self.x_i = np.clip(np.int32(self.x_i), 0, (self.processed_image_width - 1))  # To avoid out of index issues
            self.y_i = np.clip(np.int32(self.y_i), 0, (self.processed_image_height - 1))

            # because we index the gradient angle in the roh only, we subtract self.roh_ys for coordinate compatibility;
            # self.y_i are indexes on the coordinate system of the processed image.
            self.roh_grad_angle_i = self.roh_grad_angle[np.clip(self.y_i - self.roh_ys, 0, 49),
                                                        self.x_i]  # gradient angle along the i-th stretched segment
            self.SDGD_i = np.nanstd(a=self.roh_grad_angle_i, ddof=1)  # the ddof = 1 allows to divide by n - 1
            self.SDGD[index] = self.SDGD_i
            index = index + 1
            # (as in equation 4) instead of n
        survived_segs_indexes = np.where(self.SDGD < self.SDGD_th)[0]  # self.SDGD_th = 18 (based on the paper)
        if survived_segs_indexes.shape[0] == 0:
            self.detected_hl_flag = False
        else:
            self.detected_hl_flag = True
        self.survived_segs = self.roh_streched_segs[survived_segs_indexes]
        self.survived_segs_len = self.roh_streched_segs_len[survived_segs_indexes]

    def test_filtered_segments(self, src, dst, thickness=1, org=True):
        """
        This will save for each imag sample two image results: before and after segment filtering.
        :param src: src of images
        :param dst: dst of results
        :param thickness: of the line segments
        :param org: if True, the result image will be upscaled to the original size
        :return:
        """
        src_files = os.listdir(src)
        index = 0
        for src_file in src_files:
            print("{}-th image".format(index))
            img = cv.imread(os.path.join(src, src_file))
            self.start_time = time()  # to measure the computational time
            self.img_with_roh = cv.resize(img, (640, 512))
            self.img_with_segs = cv.resize(img, (640, 512))
            self.img_with_filtered_segs = cv.resize(img, (640, 512))

            self.img_input = img
            if len(self.img_input.shape) == 3:
                self.img_input_gray = cv.cvtColor(self.img_input, cv.COLOR_BGR2GRAY)
            elif len(self.img_input.shape) == 2:
                self.img_input_gray = self.img_input
            else:
                raise ValueError("Invalid shape of argument img. img must be an RGB/BGR or Grayscale image")

            self.compute_MVR()
            self.compute_Rn()
            self.get_roh_xy()
            self.get_roh()  # This method stores the ROI in self.roh
            self.extract_segments()
            self.stretch_segments()
            self.filter_segments()

            self.img_with_segs = self.draw_segments(img_dst=self.img_with_segs,
                                                    seg_set=self.roh_streched_segs,
                                                    colorlist=self.colorlist,
                                                    thickness=thickness)

            self.img_with_filtered_segs = self.draw_segments(img_dst=self.img_with_filtered_segs,
                                                             seg_set=self.survived_segs,
                                                             colorlist=self.colorlist,
                                                             thickness=thickness)
            if org:
                self.img_with_segs = cv.resize(src=self.img_with_segs, dsize=(1920, 1080))
                self.img_with_filtered_segs = cv.resize(src=self.img_with_filtered_segs, dsize=(1920, 1080))

            cv.imwrite(os.path.join(dst, "INPUT_SEGS_" + str(index) + ".png"), self.img_with_segs)
            cv.imwrite(os.path.join(dst, "OUT_SEGS_" + str(index) + ".png"), self.img_with_filtered_segs)
            # cv.imwrite(os.path.join(dst, "Segs_ROH_" + src_file), self.img_with_segs)
            self.execution_time = round(time() - self.start_time, 4)
            index = index + 1
            if index == self.samples_nbr:
                return
        return 0

    def fit_horizon(self):
        self.survived_segs_xs, self.survived_segs_ys = self.survived_segs[:, 0], self.survived_segs[:, 1]
        self.survived_segs_xe, self.survived_segs_ye = self.survived_segs[:, 2], self.survived_segs[:, 3]

        # compute segments midpoints.
        self.survived_segs_midpoint_x = np.divide(np.add(self.survived_segs_xs, self.survived_segs_xe), 2)
        self.survived_segs_midpoint_y = np.divide(np.add(self.survived_segs_ys, self.survived_segs_ye), 2)

        self.survived_segs_k = np.divide(np.subtract(self.survived_segs_ye, self.survived_segs_ys),
                                         np.subtract(self.survived_segs_xe, self.survived_segs_xs))  # slopes vector

        self.survived_segs_b = self.survived_segs_ys - (self.survived_segs_k * self.survived_segs_xs)
        self.custom_ransac()

        # print("\nMIN ERROR ACHIEVED IS:{}\n".format(self.minimum_error))
        return

    def custom_ransac(self):
        self.M = self.survived_segs_xs.shape[0]  # number of reserved segments
        self.length_weighted_error_list = np.ones(shape=(self.M,)) * 1000000
        for self.ransac_sample_index in range(0, self.M):  # we don't do sampling because the number of segments is
            # usually small. Random sampling will only increase the number of iterations with a risk of missing the true
            # horizon line (in the case where the random sampling doesn't sample its segment)
            self.compute_length_weighted_error()
            self.length_weighted_error_list[self.ransac_sample_index] = self.length_weighted_error

        self.hl_index = np.nanargmin(self.length_weighted_error_list)
        # print("index of min = ", self.hl_index)
        # print("minimum is: ", self.length_weighted_error_list[self.hl_index])
        self.hl_slope_res = self.survived_segs_k[self.hl_index]
        self.hl_intercept_res = self.survived_segs_b[self.hl_index]
        return

    def compute_length_weighted_error(self):
        """
        Computes the length weighted error of the current candidate line (this method must be plug in a loop)
        :return:
        """
        # todo find the line parameters k' and b'
        self.k = self.survived_segs_k[self.ransac_sample_index]
        self.b = self.survived_segs_b[self.ransac_sample_index]
        if isinf(self.k) or isinf(self.b):
            self.length_weighted_error = np.nan
            return
        # todo compute the distance (custom ransac error) from all midpoints to that line
        self.d = (self.survived_segs_midpoint_y - (self.k * self.survived_segs_midpoint_x) - self.b) / np.sqrt(
            (self.k ** 2) + 1)
        # print("distances = ", self.d)
        # todo weight the computed distances using segments length

        self.length_weighted_error = np.nansum(np.divide(self.d, self.survived_segs_len))  # equivalent to equation 11
        self.length_weighted_error = np.abs(self.length_weighted_error)
        # print("distance = ", np.sum(self.d))
        # print("weighted distance = ", self.length_weighted_error)
        return self.length_weighted_error, self.k, self.b

    def convert_hl_pars(self):
        self.xs_hl = int(0)
        self.xe_hl = int(640 - 1)  # int(1920 - 1)
        self.ys_hl = int(self.hl_intercept_res)  # = int((self.hl_slope * self.xs_hl) + self.hl_intercept)
        self.ye_hl = int((self.xe_hl * self.hl_slope_res) + self.hl_intercept_res)

        self.scale_height = (self.org_height - 1) / (self.res_height - 1)
        self.scale_width = (self.org_width - 1) / (self.res_width - 1)

        self.xs_hl_org = int(0)
        self.xe_hl_org = int(self.org_width - 1)  # int(1920 - 1)
        self.ys_hl_org = int(self.ys_hl * self.scale_height)  # = int((self.hl_slope * self.xs_hl) + self.hl_intercept)
        self.ye_hl_org = int(self.ye_hl * self.scale_height)

        self.hl_slope_org = (self.ye_hl_org - self.ys_hl_org) / (self.xe_hl_org - self.xs_hl_org)
        self.hl_intercept_org = self.ys_hl_org - (self.hl_slope_org * self.xs_hl_org)

        self.Y = ((((self.org_width - 1) / 2) * self.hl_slope_org + self.hl_intercept_org))
        self.phi = (-atan(self.hl_slope_org)) * (180 / pi)  # - because the y axis of images goes down

    def draw_hl(self):
        """
        Draws the horizon line on attribute 'self.img_with_hl' if it is detected. Otherwise, the text 'NO HORIZON IS
        DETECTED' is put on the image.
        """
        self.img_with_hl = np.copy(self.img_rgb_org)
        if self.detected_hl_flag:
            cv.line(self.img_with_hl, (self.xs_hl_org, self.ys_hl_org),
                    (self.xe_hl_org, self.ye_hl_org),
                    (0, 0, 255), 5)
        else:
            put_text = "NO HORIZON IS DETECTED"
            org = (int(1080 / 2), int(1920 / 2))
            color = (0, 0, 255)
            cv.putText(img=self.img_with_hl, text=put_text, org=org, fontFace=0, fontScale=2, color=color, thickness=3)

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                 draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds
        between 0 and 1) specifying the ratio of the diameter of the resized image being processed. For instance, if
        the attributre self.dsize = (640, 480), the threshold that will be used in the hough transform is sqrt(640^2 +
        480^2) * hough_threshold_ratio, rounded to the nearest integer.
        :param src_gt_folder: absolute path to the ground truth horizons corresponding to source video files.
        :param src_video_folder: absolute path to folder containing source video files to process
        :param dst_video_folder: absolute path where video files with drawn horizon will be saved.
        :param dst_quantitative_results_folder: destination folder where quantitative results will be saved.
        :param draw_and_save: if True, all detected horizons will be drawn on their corresponding frames and saved as video files
        in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            # yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
            #            " to proceed, otherwise, click on 'n'.\n"
            #            "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
            #            "files with similar names.")
            yn = 'y'
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        nbr_of_vids = len(src_video_names)
        vid_indx = 0
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            vid_indx += 1
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "Lili.Dong_" + src_video_name)
                if draw_and_save:
                    video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.org_width, self.org_height),
                                                  True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            #
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
            if nbr_of_frames != nbr_of_annotations:
                warning_text_1 = "The number of annotations (={}) does not equal to the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                print("----------WARNING---------")
                print(warning_text_1)
                print("--------------------------")

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            self.__init__()  # reinitialize all attribures of the algorithm before processing the new video
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.input_img = frame
                self.get_horizon(img=self.input_img)  # gets the horizon position and
                imgSegments = self.draw_segments(img_dst=self.img_with_roh, seg_set=self.survived_segs,
                                                 colorlist=self.colorlist)
                cv.imwrite("current_segments.png", imgSegments)
                # tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon[0], gt_horizon[1]
                print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}".
                      format(self.Y, self.gt_position_hl, self.phi, self.gt_tilt_hl))
                print("Frame {}/{}. Video {}/{}".format(idx, nbr_of_frames, vid_indx, nbr_of_vids))
                self.det_horizons_per_file[idx] = [self.Y,
                                                   self.phi,
                                                   round(abs(self.Y - self.gt_position_hl), 4),
                                                   round(abs(self.phi - self.gt_tilt_hl), 4),
                                                   self.latency]
                if draw_and_save:
                    self.img_with_segs = self.draw_segments(img_dst=self.img_with_segs,
                                                            seg_set=self.roh_streched_segs,
                                                            colorlist=self.colorlist,
                                                            thickness=1)
                    write_intermediate = False
                    if write_intermediate:
                        self.img_with_segs = cv.resize(self.img_with_segs, (self.org_width, self.org_height))
                        video_writer.write(self.img_with_segs)
                        self.img_with_filtered_segs = self.draw_segments(img_dst=self.img_with_filtered_segs,
                                                                         seg_set=self.survived_segs,
                                                                         colorlist=self.colorlist,
                                                                         thickness=1)
                        self.img_with_filtered_segs = cv.resize(self.img_with_filtered_segs,
                                                                (self.org_width, self.org_height))
                        video_writer.write(self.img_with_filtered_segs)
                    # self.img_rgb_org = self.draw_midpoints(img_dst=self.img_rgb_org)
                    self.draw_hl()  # draws the horizon on self.img_with_hl
                    video_writer.write(self.img_with_hl)
            cap.release()
            if draw_and_save:
                video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))

            # saving the .npy file of quantitative results of current video file # # # # # # # # # # # # # # # # # # # #
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                          src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                         "all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)

        self.Y_hl_all = self.det_horizons_all_files[:, 2]
        self.alpha_hl_all = self.det_horizons_all_files[:, 3]
        self.latency_all = self.det_horizons_all_files[:, 4]
        self.false_positive_nbr = np.size(np.argwhere(np.isnan(self.Y_hl_all)))

    def draw_midpoints(self, img_dst):
        colorlist = self.colorlist
        img_dst = np.float32(img_dst)
        nbr_of_colors = len(colorlist) - 1
        colorindex = 0
        for x, y in zip(self.survived_segs_midpoint_x, self.survived_segs_midpoint_y):
            cv.circle(img_dst, center=(int(x), int(y)), radius=2, thickness=-4, color=colorlist[colorindex])
            if colorindex < nbr_of_colors:
                colorindex += 1
            else:
                colorindex = 0
        return np.uint8(img_dst)
