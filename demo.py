import cv2
from skimage import morphology, img_as_ubyte
from skimage import img_as_ubyte
import numpy as np
import imutils
import math
import os
delta = {(0, 0): (-1, -1), (0, 1): (-1, 0), (0, 2): (-1, 1),
         (1, 0): (0, -1), (1, 1): (0, 0), (1, 2): (0, 1),
         (2, 0): (1, -1), (2, 1): (1, 0), (2, 2): (1, 1)}

class Font(object):
    def __init__(self):
        self.index = 0
        self.save_txt_dir = "./bihua_data/"
        self.object_name = ["星"]
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
    def threshold_image(self,image):
        ret, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    # def eight(image):
    #     def get_8count(N_P):
    #         N_P=[1-one for one in N_P]
    #         count=N_P[6]-N_P[6]*N_P[7]*N_P[0]
    #         count+=N_P[0]-N_P[0]*N_P[1]*N_P[2]
    #         count+=N_P[2]-N_P[2]*N_P[3]*N_P[4]
    #         count+=N_P[4]-N_P[4]*N_P[5]*N_P[6]
    #         return count
    #
    #     def sum_static(b_N):
    #         sum_ = 0
    #         for i, one in enumerate(b_N):
    #             if one !=-1:sum_ +=1
    #             else:
    #                 copy = one
    #                 b_N[i]=0
    #             if get_8count(b_N) == 1:
    #                 sum_ += 1
    #                 b_N[i]=copy
    #                 return sum_
    #     def neighbours(p, w, image):
    #         p_1,p1,w_1,w1=p-1,p+1,w-1,w+1
    #         # location_xy=[[p,w_1],[p_1,w],[p_1,w],]
    #         location_xy = [[p ,w_1],[p_1,w],[p_1,w],[p_1,w1],[p,w1],[p1,w1],[p1,w],[p1,w_1]]
    #         b_N=[-1 if image[x,y]==128 else if image[x,y]==2 else 0 for x,y in location_xy]
    #         return b_N
    #     def hiliditch(img_src):
    #         height,width=img_src.shape[:2];iThin=img_src.copy();th=1
    #         while th>0:
    #             th=0
    #             for p in range(1,height-1):
    #                 for w in range(1,width-1):
    #                     [x0,x1,x2,x3,x4,x5,x6,x7]=neighbours((p,w,iThin)
    #                     b_N=[x0,x1,x2,x3,x4,x5,x6,x7]
    #                     func_1=iThin[p,w]==255
    #                     func_2=1-abs(x0)+1-abs(x2)+1-abs(x4)+1-abs(x6)>=1
    #                     func_3=sum([abs(one)for one in b_N])>=2
    #                     func_4=get_8count(b_N)==1
    #                     func_5=sum([1 if one ==1 else 0 for one in b_N])>1
    #                     func_6=sum_static(b_N)==8
    #                     if func_1 and func_2 and func_3 and func_4 and func_5 and func_6:
    #                         iThin[h,w]=128;th=1;
    #                 for h in range(1,height-1):
    #                     for w in range(1,width-1):
    #                         if iThin[p,w]==128:iThin[p,w]=0
    #             return iThin
    #     def denoising(image):
    #     pass

    def getSkeleton(self,image):
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


    def draw(self,contours, image, x0=0, y0=0):
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
            center_point = [(newx + weight) // 2, (newy + height) // 2]
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

        for i in range(len(rect_contours)):
            cv2.rectangle(image, (rect_contours[i][0], rect_contours[i][1]), (rect_contours[i][2], rect_contours[i][3]),
                          255, 1)

            cv2.putText(image, str(i), (rect_contours[i][0], rect_contours[i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255, 1)
        cv2.imshow("draw", image)
        return rect_contours


    def findContours(self,image, Min_Area=10, Max_Area=8000):
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
        return temp_contours


    def strokeSplit(self,image):
        draw_image = image.copy()
        contours = self.findContours(image)
        sorted_contours = self.draw(contours, draw_image)

        # cv2.imshow("draw",draw_image)
        return sorted_contours

    def txt_add_point(self,px, py):
        add_point = '{},{}\n'.format(int(px), int(py))
        self.f.write(add_point)
    def strokeGet(self,contours, image):
        image_neighbour = image.copy()
        blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
        # sum_int=0
        for contour in contours:
            image_neighbour = image[contour[1]:contour[3], contour[0]:contour[2]].copy()
            while np.sum(image_neighbour) > 0:
                preX = preY = k = firstX = firstY = None
                for i in range(contour[0], contour[2]):
                    for j in range(contour[1], contour[3]):
                        if image_neighbour[j - contour[1], i - contour[0]] == 255:
                            if preX is None and preY is None:
                                preX, preY = i, j
                                firstX, firstY = i, j
                                blank_image[j, i] = 255
                                image_neighbour[j - contour[1], i - contour[0]] = 0
                                self.txt_add_point(0,0)
                                self.txt_add_point(i,j)
                                cv2.imshow("1", blank_image)
                                continue
                            # print(cal_distance([preX,preY],[i,j]))
                            if k is None and preX is not None and preY is not None:
                                k = self.cal_k([i-firstX, j-firstY], [0, 0])
                                print([i-firstX, j-firstY])
                                print("原来的斜率",k)
                            elif self.cal_distance([preX, preY], [i, j]) <= 5:
                                # if cal_k([preX, preY], [i, j]):
                                neighbourhood=image_neighbour[j - 1 - contour[1]: j + 2 - contour[1], i - 1- contour[0]: i + 2- contour[0]]
                                print(neighbourhood)
                                neighbours= np.argwhere(neighbourhood)
                                # print(neighbours )
                                print("计算的斜率",self.cal_k([preX-firstX, preY-firstY], [i-firstX, j-firstY]))
                                if len(neighbours)==1:
                                    image_neighbour[j - contour[1], i - contour[0]] = 0
                                elif len(neighbours)>=1 and abs(self.cal_k([preX-firstX, preY-firstY], [i-firstX, j-firstY])-k)<=k*0.10+1000:
                                    blank_image[j, i] = 255
                                    image_neighbour[j - contour[1], i - contour[0]] = 0
                                    cv2.imshow("1", blank_image)
                                    preX, preY = i, j
                                    self.txt_add_point(i,j)
                            # neighbourhood = image[j - 1: j + 2, i - 1: i + 2]
                            # neighbours = np.argwhere(neighbourhood)
                            # blank_image[j,i]=255
                            # cv2.imshow("1",blank_image)
                            # cv2.waitKey(0)
        self.f.close()
    def jiaodian(self,image):
        image = np.float32(image)
        dst = cv2.cornerHarris(image, 2, 3, 0.04)
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        image[dst > 0.02 * dst.max()] = 0
        return image

    def cal_k(self,p1, p2):
        return abs((p1[1] - p2[1]) / (p1[0] - p2[0] + 1e-5))


    def cal_distance(self,p1, p2):
        return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


    def reSize(self,image, out_height, out_width):
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
                                               dp[h - 1][w + 1][1]]) + w - 1, energy_map[h][w] + min(dp[h - 1][w - 1][1],
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

    def demo(self):
        print("第%s个字[%s]" % (self.index + 1, self.object_name[self.index]))
        print("开始采集")
        self.collect()
        print("采集结束")
        while (1):
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                self.f.close()
                break
            if (k == ord('c')) and (self.start):
                print("开始采集")
                self.collect()
                print("采集结束")
            if k == ord('n'):
                self.f.close()
                self.start = True
                if (self.index < len(self.object_name) - 1):
                    self.index += 1
                    self.next_write()

                    print("下一个字[%s]" % self.object_name[self.index])
                else:
                    print("没字了")
                    break


if __name__ == '__main__':
    font=Font()
    gary_img = cv2.imread("images/xing.png", 0)
    cv2.imshow("orginal", gary_img)
    # resize_img=reSize(gary_img,100,100)
    resize_img = cv2.resize(gary_img, (100, 100))
    cv2.imshow("resize", resize_img)
    skeleton_img = font.getSkeleton(resize_img)
    cv2.imshow("skeleton", skeleton_img)
    sorted_contour = font.strokeSplit(skeleton_img)
    # jiaodian_img=jiaodian(skeleton_img)
    # cv2.imshow("jiaodian",jiaodian_img)
    # print(jiaodian_img)
    font.strokeGet(sorted_contour, skeleton_img)
    cv2.waitKey(0)
