import numpy as np
import cv2
import functools
from shapely.geometry import Polygon
from shapely.geometry import box
from utils.plots import plot_one_box

def detect_ppe(img, box_list, hardhats, vests, masks, no_hardhats, no_vests, no_masks):
    # annotated_image = img.copy()
    flag  = False
    predicted_bboxes_PascalVOC = box_list
    if len(predicted_bboxes_PascalVOC)>0:
            for item in predicted_bboxes_PascalVOC:
                # print(item)
                name = str(item[0])
                color = (255,0,0)# blue
                if name == 'Person':
                    px0, py0, px1, py1 = item[1], item[2], item[3], item[4]
                    name = []
                    color_green = False
                    color_red = False
                    for object in predicted_bboxes_PascalVOC:
                        object_name = object[0]
                        if object_name != 'Person':
                            x0, y0, x1, y1 = object[1], object[2], object[3], object[4]
                            #print (x0,y0,x1,y1)
                            #print(px0,py0,px1,py1)
                            # print(object_name)
                            if x0 >= px0 and y0 >= py0 and x1 <= px1 and y1 <=py1:
                                # the object box is contained inside the person box
                                name.append(object_name)
                    # print(person)
                    color_green = should_color(hardhats, vests, masks, 'Hardhat','Safety Vest', 'Mask', name)
                    # print(no_hardhat,no_vest,no_mask)  
                    color_red = may_color(no_hardhats, no_vests, no_masks,'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask', name)
                    #print(color_green,color_red)
                    if color_green == True and color_red == True:
                        # something is wrong with prediction
                        color = (255,0,0)
                    else:
                        if color_green == True:
                            color = (0,255,0)
                        if color_red == True:
                            color = (0,0,255)
                    flag = color_green or color_red
                    draw_box(img,str(name),px0,py0,px1,py1,color)
    return flag

def may_color(b1, b2, b3, str1, str2, str3, items:list):
    item1, item2, item3 = False, False, False
    if str1 in items:
        item1 = True
    if str2 in items:
        item2  = True 
    if str3 in items:
        item3 = True  
    # print(hardhat,vest,mask)  
    if b1 and b2 and b3:
        return item1 or item2 or item3
    elif b1 and b2:
        return item1 or item2
    elif b1 and b3:
        return item1 or item3
    elif b2 and b3:
        return item2 or item3
    elif b1:
        return item1
    elif b2:
        return item2
    elif b3:
        return item3
    else:
        return False

def should_color(b1, b2, b3, str1, str2, str3, items:list):
    item1, item2, item3 = False, False, False
    if str1 in items:
        item1 = True
    if str2 in items:
        item2  = True 
    if str3 in items:
        item3 = True  
    # print(hardhat,vest,mask)  
    if b1 and b2 and b3:
        return item1 and item2 and item3
    elif b1 and b2:
        return item1 and item2
    elif b1 and b3:
        return item1 and item3
    elif b2 and b3:
        return item2 and item3
    elif b1:
        return item1
    elif b2:
        return item2
    elif b3:
        return item3
    else:
        return False
    
def does_overlap(x1,y1,x2,y2,x3,y3,x4,y4):
    return (x3 < x2) and (x4 > x1) and (y3 < y2) and (y4 > y1)

def draw_box(image,label,x0,y0,x1,y1,color):
    box = np.array([x0,y0,x1,y1])
    box = box.round().astype(np.int32).tolist()
    plot_one_box(box, image, label=label, color=color, line_thickness=2)

def detect_proximity(img,box_list, machines, vehicles):
    flag = False
    predicted_bboxes_PascalVOC = box_list
    if len(predicted_bboxes_PascalVOC)>0:
            for item in predicted_bboxes_PascalVOC:
                # print(item)
                name = str(item[0])
                color = (0,255,0) # green
                blue = (255,0,0)
                if name == 'Person':
                    px0, py0, px1, py1 = item[1], item[2], item[3], item[4]
                    persons = []
                    for object in predicted_bboxes_PascalVOC:
                        object_name = object[0]
                        if object_name != 'Person':
                            # print(object_name)
                            if machines and object_name == 'machinery':
                                x0, y0, x1, y1 = object[1], object[2], object[3], object[4]
                                draw_box(img,object_name,x0,y0,x1,y1,blue)
                                #print(object_name)
                                if does_overlap(x0,y0,x1,y1,px0,py0,px1,py1):
                                    # the object box overlaps with the person box
                                    persons.append(object_name)
                            if vehicles and object_name == 'vehicle':
                                x0, y0, x1, y1 = object[1], object[2], object[3], object[4]
                                draw_box(img,object_name,x0,y0,x1,y1,blue)
                                # print(object_name)
                                if does_overlap(x0,y0,x1,y1,px0,py0,px1,py1):
                                    # the object box overlaps with the person box
                                    persons.append(object_name) 
                    if len(persons) > 0:
                        color = (0,0,255)
                        flag = True
                    draw_box(img,name,px0,py0,px1,py1,color)
    return flag

def does_intersect_poly(x1,y1,x2,y2, poly):
    p = Polygon(poly)
    rect = box(x1,y1,x2,y2)
    return rect.intersects(p)

def detect_zone(img, box_list, poly, persons, machines, vehicles, inclusion, max_number_allowed):
    if poly is None:
        print('Please draw the polygon.')
        return
    predicted_bboxes_PascalVOC = box_list
    flag = False if not inclusion else True
    pts = np.array(poly)
    pts = pts.round().astype(np.int32)
    #overlay = np.zeros_like(img)
    red_color = (0,0,255)
    #cv2.fillPoly(overlay, [pts], red_color)
    #alpha = 0.25  # Transparency factor.
    # Following line overlays transparent rectangle
    # over the image
    # color_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    #cv2.imshow('Overlay',color_img)
    #cv2.imshow('Original',img)
    cv2.drawContours(img,[pts],-1,(255,0,0),2)
    t_size = cv2.getTextSize('Exclusion Zone', 0, 2/3, 2)[0]
    c1 = pts[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, (255,0,0), -1, cv2.LINE_AA)  # filled
    cv2.putText(img, 'Exclusion Zone', (c1[0], c1[1] - 2), 0, 2/3, [225, 255, 255], 2, lineType=cv2.LINE_AA)
    if len(predicted_bboxes_PascalVOC)>0:
            for item in predicted_bboxes_PascalVOC:
                #print(item)
                name = str(item[0])
                color = (0,255,0)# green
                person_count = functools.reduce(lambda x,y : x + 1 if str(y[0]) == 'Person' else x, predicted_bboxes_PascalVOC, 0)
                machine_count = functools.reduce(lambda x,y : x + 1 if str(y[0]) == 'machinery' else x, predicted_bboxes_PascalVOC, 0)
                vehicle_count = functools.reduce(lambda x,y : x + 1 if str(y[0]) == 'vehicle' else x, predicted_bboxes_PascalVOC, 0)
                if persons and name == 'Person':
                    px0, py0, px1, py1 = item[1], item[2], item[3], item[4]
                    if does_intersect_poly(px0,py0,px1,py1,poly) and person_count > max_number_allowed:
                        if not inclusion:
                            color = red_color # red
                            flag = True
                        else: 
                            flag = False
                    elif inclusion:
                        color = red_color # red
                    draw_box(img,name,px0,py0,px1,py1,color)
                if machines and name == 'machinery':
                    px0, py0, px1, py1 = item[1], item[2], item[3], item[4]
                    if does_intersect_poly(px0,py0,px1,py1,poly) and  machine_count > max_number_allowed:
                        if not inclusion:   
                            color = red_color # red
                            flag = True
                        else: 
                            flag = False
                    elif inclusion:
                        color = red_color # red
                    draw_box(img,name,px0,py0,px1,py1,color)
                if vehicles and name == 'vehicle':
                    px0, py0, px1, py1 = item[1], item[2], item[3], item[4]
                    if does_intersect_poly(px0,py0,px1,py1,poly) and vehicle_count > max_number_allowed:
                        if not inclusion:
                            color = red_color # red
                            flag = True
                        else: 
                            flag = False
                    elif inclusion:
                        color = red_color # red
                    draw_box(img,name,px0,py0,px1,py1,color)
    return flag    