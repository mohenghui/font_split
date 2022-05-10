import cv2
from skimage import morphology, img_as_ubyte
from skimage import img_as_ubyte
import numpy as np
import imutils
import math

delta = {(0, 0): (-1, -1), (0, 1): (-1, 0), (0, 2): (-1, 1),
         (1, 0): (0, -1), (1, 1): (0, 0), (1, 2): (0, 1),
         (2, 0): (1, -1), (2, 1): (1, 0), (2, 2): (1, 1)}


# 二值化
def threshold_image(image):
    ret, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold

def eight(image):
    def get_8count(N_P):
        N_P=[1-one for one in N_P]
        count=N_P[6]-N_P[6]*N_P[7]*N_P[0]
        count+=N_P[0]-N_P[0]*N_P[1]*N_P[2]
        count+=N_P[2]-N_P[2]*N_P[3]*N_P[4]
        count+=N_P[4]-N_P[4]*N_P[5]*N_P[6]
        return count

    def sum_static(b_N):
        sum_ = 0
        for i, one in enumerate(b_N):
            if one !=-1:sum_ +=1
            else:copy = one;b_N[i]=0
            if get_8count(b_N) == 1:sum_ += 1;b_N[i]=copy;return sum_


def denoising(image):
    pass

def getSkeleton(image):
    # Applies a skeletonization algorithm, thus, transforming all the strokes inside the sketch into one pixel width lines
    # threshold = threshold_image(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    threshold = threshold_image(image)
    # showImage(threshold,"opencv")
    threshold = cv2.bitwise_not(threshold)
    cv2.imshow("threshold", threshold)
    threshold[threshold == 255] = 1
    skeleton = morphology.skeletonize(threshold)  # 细化

    skeleton = img_as_ubyte(skeleton)  # 转换成8bit
    return skeleton


def draw(contours, image, x0=0, y0=0):
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
            [newx, newy, newx + weight, newy + height, center_point, cal_distance(center_point, [0, 0])])
    # rect_contours.sort(key=lambda x:x[4])
    rect_contours.sort(key=lambda x: x[4][1])
    min_y = float('inf')
    # final_rect_contours=[]
    for i in range(len(rect_contours) - 1):
        # for j in range(i,len(rect_contours)):
        distance = abs(rect_contours[i + 1][1] - rect_contours[i][1])
        print(distance)
        if distance <= 10 and rect_contours[i + 1][0] < rect_contours[i][0]:
            rect_contours[i + 1], rect_contours[i] = rect_contours[i], rect_contours[i + 1]

    for i in range(len(rect_contours)):
        cv2.rectangle(image, (rect_contours[i][0], rect_contours[i][1]), (rect_contours[i][2], rect_contours[i][3]),
                      255, 1)

        cv2.putText(image, str(i), (rect_contours[i][0], rect_contours[i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255, 1)
    cv2.imshow("draw", image)
    return rect_contours


def findContours(image, Min_Area=10, Max_Area=8000):
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


def strokeSplit(image):
    draw_image = image.copy()
    contours = findContours(image)
    sorted_contours = draw(contours, draw_image)

    # cv2.imshow("draw",draw_image)
    return sorted_contours


def strokeGet(contours, image):
    image_neighbour = image.copy()
    blank_image = np.zeros(image.shape, np.uint8)  # 做一个mask
    # sum_int=0

    for contour in contours:
        image_neighbour = image[contour[1]:contour[3], contour[0]:contour[2]].copy()
        while np.sum(image_neighbour) > 0:
            preX = preY = k = firstX = firstY = None
            print(np.sum(image_neighbour))
            for i in range(contour[0], contour[2]):
                for j in range(contour[1], contour[3]):
                    if image_neighbour[j - contour[1], i - contour[0]] == 255:
                        if preX is None and preY is None:
                            preX, preY = i, j
                            firstX, firstY = i, j
                            blank_image[j, i] = 255
                            image_neighbour[j - contour[1], i - contour[0]] = 0
                            cv2.imshow("1", blank_image)
                            continue
                        # print(cal_distance([preX,preY],[i,j]))
                        if k is None and preX is not None and preY is not None:
                            k = cal_k([preX, preY], [firstX, firstY])
                            print("原来的斜率",k)
                        elif cal_distance([preX, preY], [i, j]) <= 2:
                            # if cal_k([preX, preY], [i, j]):
                            print("计算的斜率",cal_k([preX, preY], [i, j]))
                            blank_image[j, i] = 255
                            image_neighbour[j - contour[1], i - contour[0]] = 0
                            cv2.imshow("1", blank_image)
                            preX, preY = i, j
                        # neighbourhood = image[j - 1: j + 2, i - 1: i + 2]
                        # neighbours = np.argwhere(neighbourhood)
                        # blank_image[j,i]=255
                        # cv2.imshow("1",blank_image)
                        cv2.waitKey(0)



def cal_k(p1, p2):
    return abs((p1[1] - p2[1]) / (p1[0] - p2[0] + 1e-5))


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


def reSize(image, out_height, out_width):
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


if __name__ == '__main__':
    gary_img = cv2.imread("images/hui.png", 0)
    cv2.imshow("orginal", gary_img)
    # resize_img=reSize(gary_img,100,100)
    resize_img = cv2.resize(gary_img, (100, 100))
    cv2.imshow("resize", resize_img)
    skeleton_img = getSkeleton(resize_img)
    cv2.imshow("skeleton", skeleton_img)
    sorted_contour = strokeSplit(skeleton_img)
    strokeGet(sorted_contour, skeleton_img)
    cv2.waitKey(0)
