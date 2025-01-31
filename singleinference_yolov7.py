import random
import numpy as np
import os
import sys
import torch
import cv2
import logging
from shapely.geometry import Polygon
from shapely.geometry import box

import platform
if platform.machine() == 'x86_64':
    from openvino.runtime import Model

import functools
import json
#

class SingleInference_YOLOV7:
    '''
    SimpleInference_YOLOV7
    created by Steven Smiley 2022/11/24

    INPUTS:
       VARIABLES                    TYPE    DESCRIPTION
    1. img_size,                    #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
    2. path_yolov7_weights,         #str#   #this is the path to your yolov7 weights 
    3. path_img_i,                  #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())

    OUTPUT:
       VARIABLES                    TYPE    DESCRIPTION
    1. predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)

    CREDIT
    Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
    @article{wang2022yolov7,
        title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
        author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
        journal={arXiv preprint arXiv:2207.02696},
        year={2022}
        }
    
    '''
    def __init__(self,
    img_size, path_yolov7_weights, 
    path_img_i='None',
    device_i='cpu',
    conf_thres=0.25,
    iou_thres=0.45):

        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.clicked=False
        self.img_size=img_size
        #self.path_yolov7=path_yolov7
        self.path_yolov7_weights=path_yolov7_weights
        self.path_img_i=path_img_i
        #sys.path.append(self.path_yolov7)
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        from models.experimental import attempt_load
        self.scale_coords=scale_coords
        self.non_max_suppression=non_max_suppression
        self.select_device=select_device
        self.attempt_load=attempt_load
        self.check_img_size=check_img_size
        self.open_vino_model = None

        #Initialize
        self.predicted_bboxes_PascalVOC=[]
        self.im0=None
        self.im=None
        self.device = self.select_device(device_i) #gpu 0,1,2,3 etc or '' if cpu
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.logging=logging
        #if os.path.exists('logs')==False:
        #    os.makedirs('logs')
        #self.logging.basicConfig(filename='logs/'+str(self.__class__.__name__)+'.log',filemode='w',format='%(name)s - %(levelname)s - %(message)s',level=self.logging.ERROR)
        self.logging.basicConfig(level=self.logging.DEBUG)

    def load_model(self):
        '''
        Loads the yolov7 model

        self.path_yolov7_weights = r"/example_path/my_model/best.pt"
        self.device = '0' for gpu cuda 0, '' for cpu

        '''
        # Load model
        self.model = self.attempt_load(self.path_yolov7_weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half() # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # print (self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def read_img(self,path_img_i):
        '''
        Reads a single path to a .jpg file with OpenCV

        path_img_i = r"/example_path/img_example_i.jpg"

        '''
        #Read path_img_i
        if type(path_img_i)==type('string'):
            if os.path.exists(path_img_i):
                self.path_img_i=path_img_i
                self.im0=cv2.imread(self.path_img_i)
                print('self.im0.shape',self.im0.shape)
                #self.im0=cv2.resize(self.im0,(self.img_size,self.img_size))
            else:
                log_i=f'read_img \t Bad path for path_img_i:\n {path_img_i}'
                self.logging.error(log_i)
        else:
            log_i=f'read_img \t Bad type for path_img_i\n {path_img_i}'
            self.logging.error(log_i)


    def load_cv2mat(self,im0=None):
        '''
        Loads an OpenCV matrix
        
        im0 = cv2.imread(self.path_img_i)

        '''
        if type(im0)!=type(None):
            self.im0=im0
        if type(self.im0)!=type(None):
            self.img=self.im0.copy()    
            self.imn = cv2.cvtColor(self.im0, cv2.COLOR_BGR2RGB)
            self.img=self.imn.copy()
            image = self.img.copy()
            image, self.ratio, self.dwdh = self.letterbox(image,auto=False)
            self.image_letter=image.copy()
            image = image.transpose((2, 0, 1))

            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            self.im = image.astype(np.float32)
            self.im = torch.from_numpy(self.im).to(self.device)
            self.im = self.im.half() if self.half else self.im.float()  # uint8 to fp16/32
            self.im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if self.im.ndimension() == 3:
                self.im = self.im.unsqueeze(0)
        else:
            log_i=f'load_cv2mat \t Bad self.im0\n {self.im0}'
            self.logging.error(log_i)

    def inference(self, aug=False):
        '''
        Inferences with the yolov7 model, given a valid input image (self.im)
        '''
        # Inference
        if type(self.im)!=type(None):
            
            if self.device.type == 'cpu' and self.path_yolov7_weights is None:
                self.outputs = self.open_vino_model.output(0)
                # print(self.im.shape)
                self.outputs = torch.from_numpy(self.open_vino_model(self.im)[self.outputs])
                
            else :
                if not self.path_yolov7_weights.endswith('safety.pt') and not self.path_yolov7_weights.endswith('yolov7-tiny.pt'):
                    map_file = self.path_yolov7_weights.replace(".pt", "_map.json")
                    model_object_map = json.load(open(map_file))
                else:
                    model_object_map = None
        
                with torch.no_grad():
                    self.outputs = self.model(self.im, aug)[0]
            
            # Apply NMS
            self.outputs = self.non_max_suppression(self.outputs, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
            img_i=self.im0.copy()
            self.ori_images = [img_i]
            self.predicted_bboxes_PascalVOC=[]
            for i,det in enumerate(self.outputs):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    #det[:, :4] = self.scale_coords(self.im.shape[2:], det[:, :4], self.im0.shape).round()
                    #Visualizing bounding box prediction.
                    batch_id=i
                    image = self.ori_images[int(batch_id)]

                    for j,(*bboxes,score,cls_id) in enumerate(reversed(det)):
                        x0=float(bboxes[0].cpu().detach().numpy())
                        y0=float(bboxes[1].cpu().detach().numpy())
                        x1=float(bboxes[2].cpu().detach().numpy())
                        y1=float(bboxes[3].cpu().detach().numpy())
                        self.box = np.array([x0,y0,x1,y1])
                        self.box -= np.array(self.dwdh*2)
                        self.box /= self.ratio
                        self.box = self.box.round().astype(np.int32).tolist()
                        cls_id = int(cls_id)
                        score = round(float(score),3)
                        name = self.names[cls_id]
                        
                        if model_object_map is not None:
                            label = model_object_map[name]
                            self.predicted_bboxes_PascalVOC.append([label,x0,y0,x1,y1,score]) #PascalVOC annotations
                        else :
                            self.predicted_bboxes_PascalVOC.append([name,x0,y0,x1,y1,score]) #PascalVOC annotations
        
                        color = self.colors[self.names.index(name)]
                        name += ' '+str(score)
                        cv2.rectangle(image,self.box[:2],self.box[2:],color,2)
                        # print(color)
                        cv2.putText(image,name,(self.box[0], self.box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
                    self.image=image
                else:
                    self.image=self.im0.copy()
        else:
            log_i=f'Bad type for self.im\n {self.im}'
            self.logging.error(log_i)

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        '''
        Formats the image in letterbox format for yolov7
        '''
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, r, (dw, dh)
    
    def may_color(self, b1, b2, b3, str1, str2, str3, items:list):
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
        
    def should_color(self, b1, b2, b3, str1, str2, str3, items:list):
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
 
    def detect_ppe(self, img, hardhats, vests, masks, no_hardhats, no_vests, no_masks):
        self.load_cv2mat(img)
        # print("image loaded ...")
        # self.inference(True)
        self.inference()
        # print(yolov7_detector.conf_thres)
        # annotated_image = yolov7_detector.image.copy()
        color_image = img.copy()
        # result: List[Detection] = []
        # labels = ['Hardhat', 'Safety Vest', 'Mask', 'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']
        if len(self.predicted_bboxes_PascalVOC)>0:
                for item in self.predicted_bboxes_PascalVOC:
                    # print(item)
                    name = str(item[0])
                    color = (0,0,255)# blue
                    if name == 'Person':
                        # yolov7_detector.predicted_bboxes_PascalVOC.remove(item)
                        px0, py0, px1, py1 = item[1], item[2], item[3], item[4]
                        # person = Detection(name=[], coords=[px0,py0,px1,py1])
                        name = []
                        hardhat, vest, mask = False, False, False
                        no_hardhat, no_vest, no_mask = False, False, False
                        color_green = False
                        color_red = False
                        for object in self.predicted_bboxes_PascalVOC:
                            object_name = object[0]
                            if object_name != 'Person':
                                x0, y0, x1, y1 = object[1], object[2], object[3], object[4]
                                #print (x0,y0,x1,y1)
                                #print(px0,py0,px1,py1)
                                # print(object_name)
                                if x0 >= px0 and y0 >= py0 and x1 <= px1 and y1 <=py1:
                                    # the object box is contained inside the person box
                                    # sometimes the object box is outside the person box but still coressponds to the person
                                    name.append(object_name)
                                # print(name)
                        # print(person)
                        color_green = self.should_color(hardhats,vests,masks, 'Hardhat','Safety Vest', 'Mask', name)
                        # print(no_hardhat,no_vest,no_mask)  
                        color_red = self.may_color(no_hardhats, no_vests, no_masks,'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask', name)
                        # print(name)
                        # print(color_green,color_red)
                        if color_green == True and color_red == True:
                            # something is wrong with prediction
                            color = (0,0,255)
                        else:
                            if color_green == True:
                                color = (0,255,0)
                            if color_red == True:
                                color = (255,0,0)
                        # result.append(person)
                        box = np.array([px0,py0,px1,py1])
                        box -= np.array(self.dwdh*2)
                        box /= self.ratio
                        box = box.round().astype(np.int32).tolist()
                        # print(color)
                        cv2.rectangle(color_image,box[:2],box[2:],color,4)
                    # prob = str(round(100*item[-1],2))
        #st.image(img, caption='Input Image', use_column_width=True)
        #st.write(os.listdir())	
        # print(result)
        #st.image(color_image, caption='Output Image', use_column_width=True) 	
        # st.image(annotated_image, caption='Annotated Image', use_column_width=True)
        return color_image 	
    
    def does_overlap(self, x1,y1,x2,y2,x3,y3,x4,y4):
        return (x3 < x2) and (x4 > x1) and (y3 < y2) and (y4 > y1)

    def draw_box(self, image,x0,y0,x1,y1,ratio,dwdh,color):
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        # print(color)
        cv2.rectangle(image,box[:2],box[2:],color,4)

    def detect_proximity(self, img, machines, vehicles):
        self.load_cv2mat(img)
        # print("image loaded ...")
        self.inference()
        # print(yolov7_detector.conf_thres)
        # annotated_image = yolov7_detector.image.copy()
        color_image= img.copy()
        #result: List[Detection] = []
        # labels = ['Hardhat', 'Safety Vest', 'Mask', 'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']
        if len(self.predicted_bboxes_PascalVOC)>0:
                for item in self.predicted_bboxes_PascalVOC:
                    # print(item)
                    name = str(item[0])
                    color = (0,255,0) # green
                    if name == 'Person':
                        # yolov7_detector.predicted_bboxes_PascalVOC.remove(item)
                        px0, py0, px1, py1 = item[1], item[2], item[3], item[4]
                        # person = Detection(name=[], coords=[px0,py0,px1,py1])
                        persons = []
                        for object in self.predicted_bboxes_PascalVOC:
                            object_name = object[0]
                            if object_name != 'Person':
                                # print(object_name)
                                if machines and object_name == 'machinery':
                                    x0, y0, x1, y1 = object[1], object[2], object[3], object[4]
                                    #print(object_name)
                                    self.draw_box(color_image,x0,y0,x1,y1,self.ratio,self.dwdh,(255,0,0))
                                    if self.does_overlap(x0,y0,x1,y1,px0,py0,px1,py1):
                                        # the object box overlaps with the person box
                                        persons.append(object_name)
                                if vehicles and object_name == 'vehicle':
                                    x0, y0, x1, y1 = object[1], object[2], object[3], object[4]
                                    # print(object_name)
                                    self.draw_box(color_image,x0,y0,x1,y1,self.ratio,self.dwdh,(255,0,0))
                                    if self.does_overlap(x0,y0,x1,y1,px0,py0,px1,py1):
                                        # the object box overlaps with the person box
                                        persons.append(object_name)
                        # print(person)
                        # print(no_hardhat,no_vest,no_mask)  
                        if len(persons) > 0:
                            color = (255,0,0)
                        # result.append(person)
                        # print(result)
                        self.draw_box(color_image,px0,py0,px1,py1,self.ratio,self.dwdh,color)
                    # prob = str(round(100*item[-1],2))
        #st.image(img, caption='Input Image', use_column_width=True)
        #st.write(os.listdir())	
        # print(result)
        # st.image(color_image, caption='Output Image', use_column_width=True) 	
        # st.image(annotated_image, caption='Annotated Image', use_column_width=True) 	
        return color_image
    
    def does_intersect_poly(self, x1,y1,x2,y2, poly):
        p = Polygon(poly)
        rect = box(x1,y1,x2,y2)
        #print(rect)
        #print(p)
        #print(rect.intersection(p))
        return rect.intersects(p)

    def scale_coords_box(self, x0,y0,x1,y1,ratio,dwdh):
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        x0,y0,x1,y1 = box.tolist()
        # print(x0,y0,x1,y1)
        return x0,y0,x1,y1

    def draw_box_no_scale(self,image,x0,y0,x1,y1,color):
        box = np.array([x0,y0,x1,y1])
        box = box.round().astype(np.int32).tolist()
        # print(color)
        cv2.rectangle(image,box[:2],box[2:],color,4)
    
    def detect_zone(self, img, poly, persons, machines, vehicles, inclusion, max_number_allowed):
        red_color = (255,0,0)
        self.load_cv2mat(img)
        # print("image loaded ...")
        self.inference()
        # print(yolov7_detector.conf_thres)
        # annotated_image = yolov7_detector.image.copy()
        color_image = img.copy()
        if poly is None:
            print('Please draw the polygon.')
            return
        # print(poly)
        pts = np.array(poly)
        pts = pts.round().astype(np.int32)
        #print(pts)
        # cv2.polylines(color_image, [pts], True, (255,0,0), 2)
        overlay = color_image.copy()
        # overlay = cv2.polylines(overlay, [pts], True, (255,0,0), 2)
        cv2.fillPoly(overlay, [pts], red_color)
        alpha = 0.25  # Transparency factor.
        # Following line overlays transparent rectangle
        # over the image
        color_image = cv2.addWeighted(overlay, alpha, color_image, 1 - alpha, 0)
        # result: List[Detection] = []
        if len(self.predicted_bboxes_PascalVOC)>0:
                for item in self.predicted_bboxes_PascalVOC:
                    # print(item)
                    name = str(item[0])
                    color = (0,255,0)# green
                    person_count = functools.reduce(lambda x,y : x + 1 if str(y[0]) == 'Person' else x, self.predicted_bboxes_PascalVOC, 0)
                    machine_count = functools.reduce(lambda x,y : x + 1 if str(y[0]) == 'machinery' else x, self.predicted_bboxes_PascalVOC, 0)
                    vehicle_count = functools.reduce(lambda x,y : x + 1 if str(y[0]) == 'vehicle' else x, self.predicted_bboxes_PascalVOC, 0)
                    if persons and name == 'Person':
                        px0, py0, px1, py1 = self.scale_coords_box(item[1], item[2], item[3], item[4],
                                                        self.ratio, self.dwdh)
                        if self.does_intersect_poly(px0,py0,px1,py1,poly) and person_count > max_number_allowed:
                            if not inclusion:
                                color = red_color # red
                        elif inclusion:
                            color = red_color # red
                        self.draw_box_no_scale(color_image,px0,py0,px1,py1,color)
                    if machines and name == 'machinery':
                        px0, py0, px1, py1 = self.scale_coords_box(item[1], item[2], item[3], item[4],
                                                        self.ratio, self.dwdh)
                        if self.does_intersect_poly(px0,py0,px1,py1,poly) and  machine_count > max_number_allowed:
                            if not inclusion:
                                color = red_color # red
                        elif inclusion:
                            color = red_color # red
                        self.draw_box_no_scale(color_image,px0,py0,px1,py1,color)
                    if vehicles and name == 'vehicle':
                        px0, py0, px1, py1 = self.scale_coords_box(item[1], item[2], item[3], item[4],
                                                        self.ratio,self.dwdh)
                        if self.does_intersect_poly(px0,py0,px1,py1,poly) and vehicle_count > max_number_allowed:
                            if not inclusion:
                                color = red_color # red
                        elif inclusion:
                            color = red_color # red
                        self.draw_box_no_scale(color_image,px0,py0,px1,py1,color)
                    # prob = str(round(100*item[-1],2))
        # st.image(img, caption='Input Image', use_column_width=True)
        #st.write(os.listdir())	
        # print(result)
        # st.image(color_image, caption='Output Image', use_column_width=True) 	
        # st.image(annotated_image, caption='Annotated Image', use_column_width=True) 
        return color_image

if __name__=='__main__':  

    #INPUTS
    img_size=640
    path_yolov7_weights="./weights/yolov7-construction-custom.pt"
    path_img_i=r"./img/workers-ppe-before.jpeg"

    #INITIALIZE THE app
    app=SingleInference_YOLOV7(img_size,path_yolov7_weights,path_img_i,device_i='cpu',conf_thres=0.25,iou_thres=0.5)

    #LOAD & INFERENCE
    app.load_model() #Load the yolov7 model
    app.read_img(path_img_i) #read in the jpg image from the full path, note not required if you want to load a cv2matrix instead directly
    app.load_cv2mat() #load the OpenCV matrix, note could directly feed a cv2matrix here as app.load_cv2mat(cv2matrix)
    app.inference() #make single inference
    print(f'''
    app.predicted_bboxes_PascalVOC\n
    \t name,x0,y0,x1,y1,score\n
    {app.predicted_bboxes_PascalVOC}''') 








