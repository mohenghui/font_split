import cv2
from skimage import morphology, img_as_ubyte
from skimage import img_as_ubyte
import numpy as np
import imutils
import math
import os
from shapely import geometry

delta = {(0, 0): (-1, -1), (0, 1): (-1, 0), (0, 2): (-1, 1),
         (1, 0): (0, -1), (1, 1): (0, 0), (1, 2): (0, 1),
         (2, 0): (1, -1), (2, 1): (1, 0), (2, 2): (1, 1)}


class Font(object):
    def __init__(self):
        self.index = 0
        self.save_txt_dir = "./bihua_data/"
        self.object_name = ["卧"]
        self.object_type = ".txt"
        self.image_type = ".gif"
        # self.save_dir = './pics'
        self.mode = "w"
        self.width = 100
        self.height = 100
        self.channles = 1
        self.frame_only = False
        self.start = False
        self.step = 0
        self.flag = True
        self.dir = "./image/"
        if not os.path.isdir(self.save_txt_dir):
            os.makedirs(self.save_txt_dir)
        self.f = open(self.save_txt_dir + self.object_name[self.index] + self.object_type, "w")

    # 二值化
    def threshold_image(self, image):
        ret, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def getSkeleton(self, image):
        # Applies a skeletonization algorithm, thus, transforming all the strokes inside the sketch into one pixel width lines
        # threshold = threshold_image(image)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        threshold = self.threshold_image(image)
        # showImage(threshold,"opencv")
        threshold = cv2.bitwise_not(threshold)
        cv2.imshow("threshold", threshold)
        threshold[threshold == 255] = 1
        skeleton = morphology.skeletonize(threshold)  # 细化

        skeleton = img_as_ubyte(skeleton)  # 转换成8bit
        return skeleton

    def draw1(self, contours, image):
        # print(contours)
        blank_image = image.copy()  # 做一个mask
        for i in range(len(contours)):
            cv2.rectangle(blank_image, (contours[i][0], contours[i][1]), (contours[i][2], contours[i][3]),
                          255, 1)

            cv2.putText(blank_image, str(i), (contours[i][0] - 10, contours[i][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255,
                        1)
        return blank_image
    def draw(self, contours, image, x0=0, y0=0):
        rect_contours = []
        for item in contours:
            # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            newx = x + x0
            newy = y + y0
            center_point = [(newx + newx + weight) // 2, (newy + newy + height) // 2]
            rect_contours.append(
                [newx, newy, newx + weight, newy + height, center_point, self.cal_distance(center_point, [0, 0])])
        # rect_contours.sort(key=lambda x:x[4])
        rect_contours.sort(key=lambda x: x[4][1])
        min_y = float('inf')
        # final_rect_contours=[]
        for i in range(len(rect_contours) - 1):
            # for j in range(i,len(rect_contours)):
            distance = abs(rect_contours[i + 1][4][1] - rect_contours[i][4][1])
            # print(distance)
            if distance <= 10 and rect_contours[i + 1][4][0] < rect_contours[i][4][0]:
                rect_contours[i + 1], rect_contours[i] = rect_contours[i], rect_contours[i + 1]

        # for i in range(len(rect_contours)):
        #     cv2.rectangle(image, (rect_contours[i][0], rect_contours[i][1]), (rect_contours[i][2], rect_contours[i][3]),
        #                   255, 1)
        #
        #     cv2.putText(image, str(i), (rect_contours[i][0], rect_contours[i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255,
        #                 1)
        # cv2.imshow("draw", image)
        cv2.imshow("ordraw", self.draw1(rect_contours, image))
        return rect_contours

    def findContours(self, image, Min_Area=10, Max_Area=8000):
        blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        contours = contours[1] if imutils.is_cv3() else contours[0]
        temp_contours = []
        for contour in contours:
            # 对符合面积要求的巨型装进list
            # contoursize = cv2.contourArea(contour)
            rect = cv2.boundingRect(contour)
            contoursize = rect[2] * rect[3]
            # print("面积", contoursize)
            if contoursize >= Min_Area and contoursize < Max_Area:
                temp_contours.append(contour)
            # else:
            #     image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]=0
        cv2.imshow("donie", image)
        return temp_contours

    def strokeSplit(self, image):
        draw_image = image.copy()
        contours = self.findContours(image)
        sorted_contours = self.draw(contours, draw_image)

        # cv2.imshow("draw",draw_image)
        return sorted_contours

    def GetAngle(self, point1, point2, point3):
        """
        计算两条线段之间的夹⾓
            :param line1:
            :param line2:
            :return:
            """
        dx1 = point2[0] - point1[0]
        dy1 = point2[1] - point1[1]
        dx2 = point2[0] - point3[0]
        dy2 = point2[1] - point3[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        # print(angle2)
        if angle1 * angle2 >= 0:
            insideAngle = abs(angle1 - angle2)
        else:
            insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
        insideAngle = insideAngle % 180
        return insideAngle

    def if_inPoly(self, polygon, Points):
        line = geometry.LineString(polygon)
        point = geometry.Point(Points)
        polygon = geometry.Polygon(line)
        return polygon.contains(point)

    def is_inPoly(self, contour, point):
        # return True if contour[1]<=point[0]<=contour[3] and contour[0]<=point[1]<=contour[2] else False
        return True if contour[0] - 2 <= point[0] <= contour[2] + 2 and contour[1] - 2 <= point[1] <= contour[
            3] + 2 else False
        # return True if contour[0]  <= point[0] -1 <= contour[2]  and contour[1] -1 <= point[1] <= contour[
        #     3]  else False
    def is_inPoly2(self, contour, point):
        # return True if contour[1]<=point[0]<=contour[3] and contour[0]<=point[1]<=contour[2] else False
        return True if contour[0] - 2 <= point[0] <= contour[2] + 2 and contour[1] - 2 <= point[1] <= contour[
            3] + 2 else False
    def txt_add_point(self, px, py):
        add_point = '{},{}\n'.format(int(px), int(py))
        self.f.write(add_point)

    def distanceArray(self, point_list1, point_list2):
        minDistance = float('inf')
        maxDistance = float('-inf')
        minPoint1, minPoint2 = None, None
        maxPoint1, maxPoint2 = None, None
        # print(point_list1,point_list2)
        for i in point_list1:
            # print(i)
            for j in point_list2:
                # if i==j:continue
                k = self.cal_distance(i, j)
                if minDistance > k:
                    minPoint1, minPoint2 = i, j
                    minDistance = k
                if maxDistance < k:
                    maxPoint1, maxPoint2 = i, j
                    maxDistance = k
        return minPoint1, minPoint2, maxPoint1, maxPoint2, minDistance

    def nextPoint(self, point_list1, point_list2):
        if sum(point_list1[:4]) >= sum(point_list2[:4]):
            return point_list2, point_list1
        else:
            return point_list1, point_list2

    def midPoint(self, point1, point2):
        return abs(point2[0] + point1[0]) // 2, abs(point2[1] + point2[1]) // 2

    def strokeGet2(self, contours, pointlist, image):
        blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
        # sum_int=0
        k_contours = []
        print(contours)
        print(pointlist)
        for contour in contours:
            # print(contour)
            image_neighbour = image[contour[1]:contour[3], contour[0]:contour[2]].copy()
            visit = []
            if np.sum(image_neighbour) > 0:
                for point in pointlist:
                    # print(point)
                    # if self.if_inPoly([(contour[0],contour[1]),(contour[0],contour[3]),
                    #                   (contour[2],contour[3]),(contour[2],contour[1])],point):
                    if self.is_inPoly(contour, point):
                        # print("in",contour, point)
                        # print("yes")
                        # k_contours.append([contour,self.])
                        visit.append(point)
            if len(visit) == 2:
                k_contours.append([contour, self.cal_k([0, 0], [visit[1][i]-visit[0][i] for i in range(0,len(visit[1]))])])
            # elif len(visit)<=1:
            #     continue
            else:
                # print(visit[0])
                # k_contours.append([contour,self.cal_k(visit[0], visit[1])])#暂时这样
                # k_contours.append([contour, self.cal_k([0, 0], list(set(visit[1]) - set(visit[0])))])  # 暂时这样
                k_contours.append([contour, self.cal_k([0, 0], [visit[1][i]-visit[0][i] for i in range(0,len(visit[1]))])])  # 暂时这样
        k_contours.sort(key=lambda x: x[1])
        final_contours = []
        print(k_contours)
        i = 0
        index_list=[i for i in range(len(k_contours))]
        # print(index_list)
        # for i in range(len(k_contours)):
        # while i < len(k_contours):
        while len(index_list)>=1:
            # for i in range(len(k_contours)):
            # for j in range(i+1,len(k_contours)):
            # j = i + 1
            i=index_list[0]
            index_list.remove(i)
            j = i
            tmp = k_contours[i]
            # print("og",i,tmp)
            while j < len(k_contours):
                # print(k_contours[i])
                if tmp==k_contours[j]:
                    j+=1
                    continue
                # print(self.cal_distance(k_contours[i],))
                # print(self.cal_distance(k_contours[i],
                #                         self.distanceArray([(k_contours[i][:2])],[(k_contours[i][2:4])])))
                # print(k_contours[i][:2])
                # print("tmp",tmp[0],k_contours[j][0])
                # print(self.distanceArray([k_contours[i][0][:2],k_contours[i][0][2:4]],[k_contours[j][0][:2],k_contours[j][0][2:4]])[2],abs(k_contours[i][1]-k_contours[j][1])<=k_contours[i][1]*0.10+100)
                # print(tmp,k_contours[j])
                minPoint1, minPoint2, maxPoint1, maxPoint2, minDistance = self.distanceArray([tmp[0][:2], tmp[0][2:4]],
                                                                                             [k_contours[j][0][:2],
                                                                                              k_contours[j][0][2:4]])
                # print(minPoint1, minPoint2, maxPoint1, maxPoint2, minDistance)
                # print(k_contours[i])
                if minDistance <= 10 and abs(tmp[1] - k_contours[j][1]) <= tmp[1] * 0.10 + 15 and sum(tmp[0][:2])<sum(k_contours[j][0][:2]):
                    # final_contours.append()
                    # print(1)
                    # print([i for i in maxPoint1])
                    tmp = [[maxPoint1[0], maxPoint1[1], maxPoint2[0], maxPoint2[1],
                            [zz for zz in self.midPoint(maxPoint1, maxPoint2)],
                            self.cal_distance(maxPoint1, maxPoint2)],
                           # self.cal_k([0, 0], list(set(maxPoint2) - set(maxPoint1)))]
                           self.cal_k([0, 0], [maxPoint2[i]-maxPoint1[i] for i in range(0,len(maxPoint2))])]
                    if j in index_list:index_list.remove(j)
                    j += 1
                elif minDistance <= 10 and tmp[1] >= 9999 and sum(tmp[0][:2])<sum(k_contours[j][0][:2]):
                    tmp = [[maxPoint1[0], maxPoint1[1], maxPoint2[0], maxPoint2[1],
                            [zz for zz in self.midPoint(maxPoint1, maxPoint2)],
                            # self.cal_distance(maxPoint1, maxPoint2)], self.cal_k([0, 0], list(set(maxPoint2) - set(maxPoint1)))]
                            self.cal_distance(maxPoint1, maxPoint2)], self.cal_k([0, 0], [maxPoint2[i]-maxPoint1[i] for i in range(0,len(maxPoint2))])]
                    # index_list.remove(i)
                    if j in index_list:index_list.remove(j)
                    j += 1
                else:
                    j += 1
                    # break
            if tmp!=k_contours[i]:
                # print(k_contours[i],k_contours[j-1])
                in_tmp=[tmp_i for tmp_i in tmp[0]]
                in_tmp.append(tmp[1])
                final_contours.append(in_tmp)
            else:
                in_tmp=[tmp_i for tmp_i in k_contours[i][0]]
                in_tmp.append(k_contours[i][1])
                final_contours.append(in_tmp)
        # final_contours.sort(key=lambda x:x[5])
        lost_contours=[]
        # print(contours)
        # print(final_contours)
        for i in contours:
            in_flag1=False
            in_flag2=False
            for j in final_contours:
                # print("3",j[4],i[4])
                if not self.is_inPoly2(j[:4],i[4]):
                    in_flag1=True
                    # lost_contours.append(i)
                else:in_flag2=True
            # print(in_flag1,in_flag2)
            if in_flag1 and not in_flag2:
                in_tmp=i
                in_tmp.append(self.cal_k([0, 0], [i[2]-i[0],i[3]-i[1]]))
                lost_contours.append(in_tmp)
        # print("lost",lost_contours)
        for i in lost_contours:final_contours.append(i)
        final_contours.sort(key=lambda x: x[5])
        # print(final_contours)
        return final_contours

    def intermediates(self,p1, p2, nb_points=4):
        """"Return a list of nb_points equally spaced points
        between p1 and p2"""
        # If we have 8 intermediate points, we have 8+1=9 spaces
        # between p1 and p2
        x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
        y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

        return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]
                for i in range(1, nb_points + 1)]
    def strokeGet(self, contours, image):
        # image_neighbour = image.copy()
        blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
        # sum_int=0
        print(contours)
        # print(len(contours))
        for contour in contours:
            image_neighbour = image[contour[1]:contour[3], contour[0]:contour[2]].copy()
            while np.sum(image_neighbour) > 0:
                preX = preY = None
                for j in range(contour[1], contour[3]):
                    for i in range(contour[0], contour[2]):
                        if image_neighbour[j - contour[1], i - contour[0]] == 255:
                            if preX is None and preY is None:
                                preX, preY = i, j
                                blank_image[j, i] = 255
                                image_neighbour[j - contour[1], i - contour[0]] = 0
                                self.txt_add_point(0, 0)
                                self.txt_add_point(i, j)
                                cv2.imshow("1", blank_image)
                                continue
                            tmp_dis=self.cal_distance([preX, preY], [i, j])
                            if tmp_dis <= 5:
                                if tmp_dis>=2:
                                    for bu in self.intermediates([preX,preY],[i,j]):
                                        blank_image[int(bu[1]), int(bu[0])] = 255
                                        # image_neighbour[ - contour[1], i - contour[0]] = 0
                                        cv2.imshow("1", blank_image)
                                        preX, preY = int(bu[0]), int(bu[1])
                                        self.txt_add_point(int(bu[0]), int(bu[1]))

                                neighbourhood = image_neighbour[j - 1 - contour[1]: j + 2 - contour[1],
                                                i - 1 - contour[0]: i + 2 - contour[0]]
                                neighbours = np.argwhere(neighbourhood)
                                if len(neighbours) == 1:
                                    image_neighbour[j - contour[1], i - contour[0]] = 0
                                else:
                                    blank_image[j, i] = 255
                                    image_neighbour[j - contour[1], i - contour[0]] = 0
                                    cv2.imshow("1", blank_image)
                                    preX, preY = i, j
                                    self.txt_add_point(i, j)


                            # cv2.waitKey(0)
        self.f.close()

    def jiaodian(self, image):
        contour = []
        gray = np.float32(image)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        dst = dst > 0.02 * dst.max()
        # dst = cv2.goodFeaturesToTrack(gray, 80, 0.05, 10)
        blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
        # Threshold for an optimal value, it may vary depending on the image.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if dst[i][j]:
                    contour.append([i, j])
                    blank_image[i][j] = 255
        # dst = (dst.astype(int)).tolist()
        # for z in dst:
        #     contour.append([z[0][0],z[0][1]])
        #     blank_image[z[0][1]][z[0][0]]=255
        return blank_image, contour

    def cal_k(self, p1, p2):
        return abs((p1[1] - p2[1]) / (p1[0] - p2[0] + 1e-5))

    def cal_distance(self, p1, p2):
        return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

    def reSize(self, image, out_height, out_width):
        height_input, width_input = image.shape[:2]
        out_image = np.copy(image)
        delta_row, delta_col = abs(int(out_height - height_input)), abs(int(out_width - width_input))
        row_turn = 0

        for each_dimen in ['col', 'row']:
            if each_dimen == 'row':
                out_image = np.rot90(out_image)
                delta = delta_row
                row_turn += 1
            else:
                delta = delta_col

            if delta > 0:
                for each_iter in range(delta):
                    if len(out_image.shape) > 2:
                        b, g, r = cv2.split(out_image)
                        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
                        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
                        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
                        energy_map = r_energy + g_energy + b_energy
                    # else:
                    #     energy = cv2.split(out_image)
                    #     energy = np.absolute(cv2.Scharr(energy, -1, 1, 0)) + np.absolute(cv2.Scharr(energy, -1, 0, 1))
                    #     energy_map = energy
                    height, width = energy_map.shape

                    dp = [[(None, 0) for i in range(width)] for j in range(height)]
                    dp[0] = [(None, 1) for i in energy_map[0]]

                    for h in range(1, height):
                        for w in range(width):
                            if w == 0:
                                dp[h][w] = (
                                    np.argmin([dp[h - 1][w][1], dp[h - 1][w + 1][1]])
                                    + w, energy_map[h][w] + min(dp[h - 1][w][1],
                                                                dp[h - 1][w + 1][1]
                                                                ))
                            elif w == width - 1:
                                dp[h][w] = (
                                    np.argmin([dp[h - 1][w - 1][1], dp[h - 1][w][1]])
                                    + w - 1, energy_map[h][w] + min(dp[h - 1][w - 1][1],
                                                                    dp[h - 1][w][1]
                                                                    ))
                            else:
                                dp[h][w] = (
                                    np.argmin([dp[h - 1][w - 1][1], dp[h - 1][w][1],
                                               dp[h - 1][w + 1][1]]) + w - 1,
                                    energy_map[h][w] + min(dp[h - 1][w - 1][1],
                                                           dp[h - 1][w][1],
                                                           dp[h - 1][w + 1][1]
                                                           ))

                    backtrace = []
                    cur = np.argmin([i[1] for i in dp[-1]])
                    backtrace.append(cur)
                    row = height - 1
                    while cur is not None:
                        cur = dp[row][cur][0]
                        backtrace.append(cur)
                        row -= 1

                    min_energy_idx = backtrace[:-1][::-1]
                    m, n = out_image.shape[:2]
                    output = np.zeros((m, n - 1, 3))

                    for row in range(m):
                        col = min_energy_idx[row]
                        output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
                        output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
                        output[row, :, 2] = np.delete(out_image[row, :, 2], [col])
                    out_image = np.copy(output)

        if row_turn == 1:
            out_image = np.rot90(out_image, 3)

        return out_image

    def secondTiny(self, contours, image, x0=0, y0=0):
        rect_contours = []
        for item in contours:
            # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            newx = x + x0
            newy = y + y0
            center_point = [(newx + newx + weight) // 2, (newy + newy + height) // 2]
            image[center_point[1], center_point[0]] = 255
            rect_contours.append(center_point)
        return rect_contours, image

    def jiaoGetSkeleton(self, image, skimage):
        # Applies a skeletonization algorithm, thus, transforming all the strokes inside the sketch into one pixel width lines
        # threshold = threshold_image(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.erode(image, kernel)
        # cv2.imshow("123",eroded)
        contour = self.findContours(image, 0, 100)
        contour, image = self.secondTiny(contour, blank_image)
        cv2.imshow("imgg", image)
        for i in contour:
            skimage[i[1] - 1:i[1] + 1, i[0] - 1:i[0] + 1] = 0
            # skimage[i[1] - 2, i[0] - 2] = 0
            # skimage[i[1] + 2, i[0] + 2] = 0
            # skimage[i[1] - 2, i[0] + 2] = 0
            # skimage[i[1] + 2, i[0] - 2] = 0
        return skimage, contour

    def denoe(self, image, contours):
        blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
        # sum_int=0
        new_contours = []
        for contour in contours:
            image_neighbour = image[contour[1]:contour[3], contour[0]:contour[2]].copy()
            if np.sum(image_neighbour) >= 255:
                # print(np.sum(image_neighbour))
                new_contours.append(contour)
        return new_contours

    def demo(self):
        print("第%s个字[%s]" % (self.index + 1, self.object_name[self.index]))
        print("开始采集")
        # self.collect()
        print("采集结束")
        while (1):
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                self.f.close()
                break


if __name__ == '__main__':
    font = Font()
    gary_img = cv2.imread("images/000003.png", 0)
    cv2.imshow("orginal", gary_img)
    # resize_img=reSize(gary_img,100,100)
    resize_img = cv2.resize(gary_img, (150, 150))
    cv2.imshow("resize", resize_img)
    skeleton_img = font.getSkeleton(resize_img)
    cv2.imshow("skeleton", skeleton_img)
    img1, contour = font.jiaodian(skeleton_img)
    cv2.imshow("ogjiaodian", img1)
    # sorted_contour = font.strokeSplit(skeleton_img)
    jiaodian_img, sort_contour = font.jiaoGetSkeleton(img1, skeleton_img)
    # jiaodian_img=jiaodian(skeleton_img)
    cv2.imshow("opjiaodian", jiaodian_img)
    # print(jiaodian_img)

    sorted_contour = font.strokeSplit(jiaodian_img)
    # print(sorted_contour)
    final_contour = font.strokeGet2(sorted_contour, sort_contour, jiaodian_img)
    # print(final_contour)
    cv2.imshow("opdraw", font.draw1(final_contour, skeleton_img))
    # cv2.imshow("o", skeleton_img)
    # font.denoe(jiaodian_img,sorted_contour)
    font.strokeGet(final_contour,skeleton_img)
    cv2.waitKey(0)
