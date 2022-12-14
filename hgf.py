import cv2
import numpy as np

img = cv2.imread("public_test_images/maze_12.png")

def find_colour(img,colour):
    if (img[5][10] == colour).all():#BGR
        return True
def warp(img,squares):
    #As the coordinates are not arranged properly we reaarange them properly by checking the sum and difference of x and y coordinates
    pts = np.squeeze(squares)    
    box_width = np.max(pts[:, 0]) - np.min(pts[:, 0])    
    box_height = np.max(pts[:, 1]) - np.min(pts[:, 1])
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)
    bounding_rect = np.array([pts[np.argmin(sum_pts)],
                    pts[np.argmin(diff_pts)],
                    pts[np.argmax(sum_pts)],
                    pts[np.argmax(diff_pts)]], dtype=np.float32)
    warped = np.array([[0, 0],
            [box_width - 1, 0],
            [box_width - 1, box_height - 1],
            [0, box_height - 1]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(bounding_rect, warped)
    warped_img = cv2.warpPerspective(img, transform_matrix, (box_width, box_height))
    return [warped_img,bounding_rect,warped]


first_node = np.array([[[94,94]],[[94,106]],[[106,94]],[[106,106]]])


def node_coordiantes(first_node):
    square = []
    
    for i in range(7):
        for j in range(7):
            square.append(first_node+np.array([[[100*i,100*j]],[[100*i,100*j]],[[100*i,100*j]],[[100*i,100*j]]]))
    return square
def horizontal_roads(first_node):
    square = node_coordiantes(first_node)
    horizontal_roads = []
    for i in range(0,42):
        horizontal_roads.append([square[i][2],square[i][3],square[i+7][0],square[i+7][1]])
    
    return np.array(horizontal_roads)
def vertical_roads(first_node):
    square = node_coordiantes(first_node)
    vertical_roads = []
    for i in range(0,48):
        if (i+1)%7 != 0 :
            vertical_roads.append([square[i][1],square[i+1][0],square[i][3],square[i+1][2]])
    
    return np.array(vertical_roads)


address = ["A1","A2","A3","A4","A5","A6","A7","B1","B2","B3","B4","B5","B6","B7","C1","C2","C3","C4","C5","C6","C7","D1","D2","D3","D4","D5","D6","D7","E1","E2","E3","E4","E5","E6","E7","F1","F2","F3","F4","F5","F6","F7","G1","G2","G3","G4","G5","G6","G7"]
h_roads = ["A1-B1","A2-B2","A3-B3","A4-B4","A5-B5","A6-B6","A7-B7","B1-C1","B2-C2","B3-C3","B4-C4","B5-C5","B6-C6","B7-C7","C1-D1","C2-D2","C3-D3","C4-D4","C5-D5","C6-D6","C7-D7","D1-E1","D2-E2","D3-E3","D4-E4","D5-E5","D6-E6","D7-E7","E1-F1","E2-F2","E3-F3","E4-F4","E5-F5","E6-F6","E7-F7","F1-G1","F2-G2","F3-G3","F4-G4","F5-G5","F6-G6","F7-G7"]
v_roads = ["A1-A2","A2-A3","A3-A4","A4-A5","A5-A6","A6-A7","B1-B2","B2-B3","B3-B4","B4-B5","B5-B6","B6-B7","C1-C2","C2-C3","C3-C4","C4-C5","C5-C6","C6-C7","D1-D2","D2-D3","D3-D4","D4-D5","D5-D6","D6-D7","E1-E2","E2-E3","E3-E4","E4-E5","E5-E6","E6-E7","F1-F2","F2-F3","F3-F4","F4-F5","F5-F6","F6-F7","G1-G2","G2-G3","G3-G4","G4-G5","G5-G6","G6-G7"]




def detect_traffic_signals(img):
    nodes = []
    traffic_signals = []
    for i in range(len(node_coordiantes(first_node))):
        nodes.append(warp(img,node_coordiantes(first_node)[i])[0])
        if find_colour(nodes[i],[0,0,255]) == True:
            traffic_signals.append(address[i])
    return {"traffic_signals":traffic_signals}
def detect_horizontal_roads_under_construction(img):
    nodes = []
    horizontal_roads_ = []
    for i in range(len(horizontal_roads(first_node))):
        nodes.append(warp(img,horizontal_roads(first_node)[i])[0])
        if find_colour(nodes[i],[255,255,255]) == True:
            horizontal_roads_.append(h_roads[i])
    return {"horizontal_roads_under_construction":horizontal_roads_}
def detect_vertical_roads_under_construction(img):
    nodes = []
    vertical_roads_ = []
    for i in range(len(vertical_roads(first_node))):
        nodes.append(warp(img,vertical_roads(first_node)[i])[0])
        if find_colour(nodes[i],[255,255,255]) == True:
            vertical_roads_.append(v_roads[i])
    return {"vertical_roads_under_construction":vertical_roads_}
def detect_medicine_packages(img):
    return
def detect_arena_parameters(img):
    D1 = detect_traffic_signals(img)
    D2 = detect_horizontal_roads_under_construction(img)
    D3 = detect_vertical_roads_under_construction(img)
    D4 = detect_medicine_packages(img)
    D5={**D1,**D2,**D3}
    return print("Arena parameters:",D5)

detect_arena_parameters(img)









#cv2.imshow("hxfgv",warp(img,vertical_roads(first_node)[39])[0])
#cv2.waitKey(0)
cv2.destroyAllWindows()