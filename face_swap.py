# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:33:59 2021

@author: mobileprogramming
"""

import cv2
import numpy as np
import dlib
import time
import mediapipe as mp
import os

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results,mp_drawing,mp_holistic):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 

def face_swap_dlib(image1,image2):
    img_src = cv2.imread(image1, cv2.IMREAD_COLOR)
    img_dst = cv2.imread(image2, cv2.IMREAD_COLOR)
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_src_gray)
    img_dst_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    height, width, channels = img_dst.shape
    img_dst_new_face = np.zeros((height, width, channels), np.uint8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(f"{os.getcwd()}\\shape_predictor_68_face_landmarks.dat")
    
    faces = detector(img_src_gray)
    if not faces:
        return img_dst,img_dst
    for face in faces:
        landmarks = predictor(img_src_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
    
            #cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        
        cv2.fillConvexPoly(mask, convexhull, 255)
    
        face_image_1 = cv2.bitwise_and(img_src, img_src, mask=mask)
    
        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        
        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
    
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)
    
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)
    
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)
    
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)
                
    faces2 = detector(img_dst_gray)
    if not faces2:
        return img_dst,img_dst
    for face in faces2:
        landmarks = predictor(img_dst_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))
    
    
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)
    
    lines_space_mask = np.zeros_like(img_src_gray)
    lines_space_new_face = np.zeros_like(img_dst)
    
    
    
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
    
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
     
        cropped_triangle = img_src[y: y + h, x: x + w]
        
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
    
    
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
       
    
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    
        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img_src, img_src, mask=lines_space_mask)
    
        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    
    
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
    
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
    
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
      
    
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    
        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
        
    
        # Reconstructing destination face
        img_dst_new_face_rect_area = img_dst_new_face[y: y + h, x: x + w]
        img_dst_new_face_rect_area_gray = cv2.cvtColor(img_dst_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img_dst_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        
        if warped_triangle.shape[0:2] != mask_triangles_designed.shape:
            warped_triangle = cv2.resize(warped_triangle,(mask_triangles_designed.shape[1],mask_triangles_designed.shape[0]))
            
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        
    
        img_dst_new_face_rect_area = cv2.add(img_dst_new_face_rect_area, warped_triangle)
        img_dst_new_face[y: y + h, x: x + w] = img_dst_new_face_rect_area
        
    img_dst_face_mask = np.zeros_like(img_dst_gray)
    img_dst_head_mask = cv2.fillConvexPoly(img_dst_face_mask, convexhull2, 255)
    img_dst_face_mask = cv2.bitwise_not(img_dst_head_mask)
    
    
    img_dst_head_noface = cv2.bitwise_and(img_dst, img_dst, mask=img_dst_face_mask)
    result = cv2.add(img_dst_head_noface, img_dst_new_face)
    
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    
    
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    
    seamlessclone = cv2.seamlessClone(result, img_dst, img_dst_head_mask, center_face2, cv2.NORMAL_CLONE)
    def ReturnName(image1, image2):
        return os.path.splitext(os.path.basename(image1))[0], os.path.splitext(os.path.basename(image2))[0]

    image1, image2 = ReturnName(image1, image2)
    
    
    path = f"{os.getcwd()}\\static\\output\\{str(image1)}-{str(image2)}.jpg"  # Generates a random path
    print(os.getcwd())
    cv2.imwrite(path, seamlessclone)  # saves the image to the path
    print("Output added to path: " + path)


def face_swap_mediapipe(image1,image2):
    img_src = cv2.imread(image1, cv2.IMREAD_COLOR)
    img_dst = cv2.imread(image2, cv2.IMREAD_COLOR)
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_src_gray)
    img_dst_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    height, width, channels = img_dst.shape
    img_dst_new_face = np.zeros((height, width, channels), np.uint8)
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    holistic = mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.5)
    
    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.5) as holistic:
        # Make detections
        img1, results1 = mediapipe_detection(img_src, holistic)
    
    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.5) as holistic:
        # Make detections
        img2, results2 = mediapipe_detection(img_dst, holistic)
    
    
    height, width, channels = img_src.shape
    landmarks_points = [(int(res.x*width), int(res.y*height)) for res in results1.face_landmarks.landmark]
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    
    face_image_1 = cv2.bitwise_and(img_src, img_src, mask=mask)
    
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
    
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
    
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
    
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
    
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
                    
    
    height, width, channels = img_dst.shape
    landmarks_points2 = [(res.x*width, res.y*height) for res in results2.face_landmarks.landmark]
    points2 = np.array(landmarks_points2, np.int32)
    convexhull2 = cv2.convexHull(points2)
    lines_space_mask = np.zeros_like(img_src_gray)
    lines_space_new_face = np.zeros_like(img_dst)
    
    
    
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
    
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
     
        cropped_triangle = img_src[y: y + h, x: x + w]
        
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
    
    
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
       
    
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    
        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img_src, img_src, mask=lines_space_mask)
    
        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    
    
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
    
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
    
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
      
    
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    
        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
        
    
        # Reconstructing destination face
        img_dst_new_face_rect_area = img_dst_new_face[y: y + h, x: x + w]
        img_dst_new_face_rect_area_gray = cv2.cvtColor(img_dst_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img_dst_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        
        if warped_triangle.shape[0:2] != mask_triangles_designed.shape:
            warped_triangle = cv2.resize(warped_triangle,(mask_triangles_designed.shape[1],mask_triangles_designed.shape[0]))
            
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        
    
        img_dst_new_face_rect_area = cv2.add(img_dst_new_face_rect_area, warped_triangle)
        img_dst_new_face[y: y + h, x: x + w] = img_dst_new_face_rect_area
        
    img_dst_face_mask = np.zeros_like(img_dst_gray)
    img_dst_head_mask = cv2.fillConvexPoly(img_dst_face_mask, convexhull2, 255)
    img_dst_face_mask = cv2.bitwise_not(img_dst_head_mask)
    
    
    img_dst_head_noface = cv2.bitwise_and(img_dst, img_dst, mask=img_dst_face_mask)
    result = cv2.add(img_dst_head_noface, img_dst_new_face)
    
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    
    
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    
    seamlessclone = cv2.seamlessClone(result, img_dst, img_dst_head_mask, center_face2, cv2.NORMAL_CLONE)
    def ReturnName(image1, image2):
        return os.path.splitext(os.path.basename(image1))[0], os.path.splitext(os.path.basename(image2))[0]

    image1, image2 = ReturnName(image1, image2)
    
    
    path = f"{os.getcwd()}\\static\\output\\{str(image1)}-{str(image2)}.jpg"  # Generates a random path
    print(os.getcwd())
    cv2.imwrite(path, seamlessclone)  # saves the image to the path
    print("Output added to path: " + path)





