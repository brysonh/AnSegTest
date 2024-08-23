import pyglet
import ctypes
import math
import numpy as np
import os
import shutil
import sys
import time
import pickle
import argparse
import datetime
import importlib
import warnings
from collections import defaultdict
from operator import itemgetter
from pyglet.gl import *
from pyglet.window import mouse
import pydicom
from scipy import ndimage
from scipy.linalg import lstsq
import networkx as nx


#Constants
C_VIEW_FOV = 45
C_VIEW_START_DIST = 3
C_ZOOM_STEP = .05
C_MIN_WINDOW_SIZE_X = 10
C_MIN_WINDOW_SIZE_Y = 10
C_ROTATION_STEP = 0.01
C_AN_OFFSET = 0.02
C_LINE_WIDTH = 5
C_POINT_SIZE = 10
C_PAN_SENSITIVITY = 0.1
C_OBJ_PATH = ('mesh/', '_full.obj')
C_AN_PATH = ('mesh_ad/', '-_norm.ad')
C_VOXEL_PATH = ('voxel/', '.dcm')
C_VOXEL_AN_PATH = ('voxel_ad/', '.ad')
C_CACHE_PATH = 'mesh/area_cache.txt'
C_MESH_CACHE_PATH = (C_OBJ_PATH[0] + 'Cache/', '.pkl')
C_VOXEL_CACHE_PATH = (C_VOXEL_PATH[0] + 'Cache/', '.pkl')
C_CALC_CACHE_PATH = ('calcTemp/','.pkl')
C_OUTPUT_FILE_DEFAULT = 'output/out_' + datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S') + '.csv'
C_OUTPUT_FILE_ROUNDING = 5
C_FORCE_RECALC = False
C_GUI_PATH = 'images/'
C_GUI_BORDER_OFFSET = 20
C_GUI_ARROW_SPACING = 20
C_GUI_ELEMENT_SPACING = 20
C_GUI_TEXT_COLOR = (255,255,255,255) #White
C_GUI_TEXT_TITLE_SIZE = 32
C_GUI_TEXT_SIZE = 24
C_GUI_TEXT_SPACING = 2
C_GUI_BACKGROUND_COLOR = (0,0,0) #Black
C_OBJ_COLOR = (255,255,255) #White
C_PATH_ACTUAL_COLOR = (0,0,255) #Blue
C_PATH_CALC_COLOR = (255,0,0) #Red
C_PATH_OVERLAP_COLOR = (0,255,0) #Green
C_SURFACE_ACTUAL_COLOR = (0, 153, 255) #Light blue
C_SURFACE_CALC_COLOR = (255, 140, 140) #Light red
C_SURFACE_OVERLAP_COLOR = (145, 250, 169) #Light green
C_VOXEL_SIZE = (0.469,0.469,0.469)
C_SURF_MAX_RECUR = 500
C_MIN_LOOP_SIZE = 10
C_MAX_PATH_ATTEMPTS = 100000


#Data structure for storing 3D coordinates
class Point():
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'


#Data structure for storing file information
class AnData():
    def __init__(self,vertices,an_vertices):
        
        #Calculate segment bounding sphere          
        xmin = min(vertices[0::3])
        xmax = max(vertices[0::3])
        ymin = min(vertices[1::3])
        ymax = max(vertices[1::3])
        zmin = min(vertices[2::3])
        zmax = max(vertices[2::3])
        
        #Calculate aneurysm bounding sphere          
        xminAn = min(an_vertices[0::3])
        xmaxAn = max(an_vertices[0::3])
        yminAn = min(an_vertices[1::3])
        ymaxAn = max(an_vertices[1::3])
        zminAn = min(an_vertices[2::3])
        zmaxAn = max(an_vertices[2::3])
                
        self.center = Point((xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2)
        self.radius = math.sqrt((xmax-self.center.x)**2+(ymax-self.center.y)**2+(zmax-self.center.z)**2)
        self.vesselDim = Point(xmax-xmin,ymax-ymin,zmax-zmin)
        self.anDim = Point(xmaxAn-xminAn,ymaxAn-yminAn,zmaxAn-zminAn)
        self.vesselBounds = (xmax,xmin,ymax,ymin,zmax,zmin)
        self.anBounds = (xmaxAn,xminAn,ymaxAn,yminAn,zmaxAn,zminAn)
      
      
#Data structure for storing file information for a mesh object
class AnDataMesh(AnData):
    def __init__(self,filename,vertices,faces,normals,an_boundary,an_vertices):
    
        self.filename = filename
        self.vertices = np.array(vertices)
        self.an_vertices = np.array(an_vertices)
        self.faces = np.array(faces)
        self.normals = np.array(normals)
        self.an_boundary = np.array(an_boundary)
        self.an_start_point = farthestPoint(np.reshape(an_boundary,(-1,3)),np.reshape(an_vertices,(-1,3)))
        self.numVertices = int(len(vertices)/3)
        self.numAnPoints = int(len(an_boundary)/3)
        self.numAnVertices = int(len(an_vertices)/3)
        self.colors = C_OBJ_COLOR*self.numVertices
        
        (self.displayVertices,self.displayNormals) = displayMesh(self.vertices,self.faces,self.normals)
        self.displayNumVertices = int(len(self.displayVertices)/3)
        self.displayColors = C_OBJ_COLOR*self.displayNumVertices
        self.displayDict = getDisplayDictMesh(self.vertices,self.displayVertices)
        
        self.graph = meshToGraph(self.vertices,self.faces)
        recolorVerticesOr(self,self.an_vertices,C_SURFACE_ACTUAL_COLOR)
        
        self.objVertexList = None
        self.anVertexList = None
        
        self.calc_boundary = []
        
        super().__init__(self.vertices,self.an_vertices)


#Data structure for storing file information for a voxel object
class AnDataVoxel(AnData):
    def __init__(self,filename,vertices,faces,an_boundary,an_vertices,displayVertices,displayFaces,displayNormals,displayDict,voxelArray):
    
        self.filename = filename
        self.vertices = np.array(vertices)
        self.an_vertices = np.array(an_vertices)
        self.faces = np.array(faces)
        self.an_boundary = np.array(an_boundary)
        self.an_start_point = vertexToVoxel(voxelArray,farthestPoint(np.reshape(an_boundary,(-1,3)),np.reshape(an_vertices,(-1,3))))
        self.numVertices = int(len(vertices)/3)
        self.numAnPoints = int(len(an_boundary)/3)
        self.numAnVertices = int(len(an_vertices)/3)
        self.colors = C_OBJ_COLOR*self.numVertices
        
        self.displayVertices = displayVertices
        self.displayFaces = displayFaces
        self.displayNormals = displayNormals
        self.displayDict = displayDict
        self.displayNumVertices = int(len(self.displayVertices)/3)
        self.displayColors = C_OBJ_COLOR*self.displayNumVertices
        
        self.graph = meshToGraph(self.vertices,self.faces)
        
        self.voxelArray = voxelArray
        (self.an_voxelArray6,self.an_voxelArray26) = getVoxelAnMask(self,self.an_vertices,self.an_boundary)
        recolorVoxelMask(self,self.an_voxelArray26,C_SURFACE_ACTUAL_COLOR)
        
        self.objVertexList = None
        self.anVertexList = None
        
        self.calc_boundary = []
        
        super().__init__(self.vertices,self.an_vertices)


#Parameterize a given path with a specified number of points
def parameterizePath(path,numPts):

    #Find the length of the path
    lenPath = 0
    lenList = []
    for i in range(len(path)-1):
        d = math.dist(path[i],path[i+1])
        lenPath += d
        lenList.append(d)
    d = math.dist(path[-1],path[0])
    lenPath += d
    lenList.append(d)
    
    #Distribute points evenly along path
    param = np.empty([numPts,3])
    step = lenPath/numPts
    curLen = 0
    curSeg = 0
    curSegLen = 0
    curVec = (path[1]-path[0])/lenList[0]
    curPoint = path[0]
    for i in range(numPts):
        param[i] = curPoint
        curLen += step
        curSegLen += step
        if (curSegLen >= lenList[curSeg]) and (i<numPts-1):
            curPoint = curPoint + curVec*(step - (curSegLen-lenList[curSeg]))
            curSegLen = curSegLen - lenList[curSeg]
            curSeg += 1
            curVec = (path[(curSeg+1)%len(path)]-path[curSeg])/lenList[curSeg]
            curPoint = curPoint + curVec*curSegLen
        else:
            curPoint = curPoint + curVec*step

    return param
    
    
#Calculate the area of a triangle given three side lengths
def triangleArea(a, b, c):   
    s = (a+b+c)/2
    a2 = s*(s-a)*(s-b)*(s-c)
    if a2<=0:
        return 0
    return math.sqrt(a2)
    

#Calculate the area of a quadrilateral given its corner points in 3D    
def quadArea(p1, p2, p3, p4):
    s1 = math.dist(p2,p1)
    s2 = math.dist(p3,p2)
    s3 = math.dist(p4,p3)
    s4 = math.dist(p1,p4)
    diag = math.dist(p1,p3)
    return triangleArea(s1,s2,diag) + triangleArea(diag,s3,s4)
    
    
#Calculate area of a triange given coordinates of its corners
def triangleAreaCoords(p1, p2, p3):
    s1 = math.dist(p2,p1)
    s2 = math.dist(p3,p2)
    s3 = math.dist(p1,p3)
    return triangleArea(s1,s2,s3)
    
    
#Quantify the error between a calculated path and the actual path   
def pathError(pathCalc, pathActual, numPts=100):

    #Match calculated starting point to actual starting point
    minDist = float('inf')
    startIndex = 0
    for i in range(len(pathCalc)):
        d = math.dist(pathCalc[i],pathActual[0])
        if d < minDist:
            minDist = d
            startIndex = i

    if startIndex != 0:
        pathCalc = np.concatenate((pathCalc[startIndex:],pathCalc[0:startIndex]))
        
    #Ensure both paths are oriented in the same direction (clockwise or counterclockwise)
    vecActual = pathActual[math.ceil(len(pathActual)/8)] - pathActual[0]
    vecCalc = pathCalc[math.ceil(len(pathCalc)/8)] - pathCalc[0]
    vecCalcInv = pathCalc[-1*math.ceil(len(pathCalc)/8)] - pathCalc[0]
    if np.dot(vecActual,vecCalcInv) > np.dot(vecActual,vecCalc):
        pathCalc[1:] = pathCalc[:0:-1]

    #Parameterize each path
    paramCalc = parameterizePath(pathCalc,numPts)
    paramActual = parameterizePath(pathActual,numPts)
    
    #Calculate area of error surface
    area = 0
    for i in range(numPts-1):
        area += quadArea(paramActual[i], paramActual[i+1], paramCalc[i], paramCalc[i+1])
    area += quadArea(paramActual[-1], paramActual[0], paramCalc[-1], paramCalc[0])
    return area


#Sort list of points by distance
def sortPath(points):
    sortedPoints = points[0:3]
    remainingPoints = points[3:]
    while(len(sortedPoints)!=len(points)):
        minDist = float('inf')
        currIndex = 0
        for i in range(0,len(remainingPoints),3):
            dist = math.dist(remainingPoints[i:i+3],sortedPoints[-3:])
            if dist<minDist:
                currIndex = i
                minDist = dist
        sortedPoints.extend(remainingPoints[currIndex:currIndex+3])
        del remainingPoints[currIndex:currIndex+3]
    return sortedPoints
    
    
#Calculate surface area of an aneurysm
def aneurysmArea(filename):
    #Create file paths
    objPath = C_OBJ_PATH[0] + filename + C_OBJ_PATH[1]
    anPath = C_AN_PATH[0] + filename + C_AN_PATH[1]
    cachePath = C_CACHE_PATH

    #Check if the surface area has already been calculated
    try:
        cache = open(cachePath)
        for line in cache:
            values = line.split()
            if values[0] == filename:
                cache.close()
                return values[1]
        cache.close()
    except FileNotFoundError:
        pass
                
    #Calculate area if it has not been calculated already
    try:
        an = open(anPath)
    except FileNotFoundError:
        return -1 
        
    keepPoints = []
    for line in an:
        values = line.split()
        if values[-1]=='1' or values[-1]=='2':
            keepPoints.append([float(values[0]),float(values[1]),float(values[2])])
        else: 
            keepPoints.append(False)  
    an.close()
    
    try:
        obj = open(objPath)
    except FileNotFoundError:
        return -1
    area = 0
    for line in obj:
        values = line.split()
        if len(values)>0 and values[0] == 'f':
            curFace = [int(values[1].split('//')[0])-1,int(values[2].split('//')[0])-1,int(values[3].split('//')[0])-1]
            if keepPoints[curFace[0]] != False and keepPoints[curFace[1]] != False and keepPoints[curFace[2]] != False:
                s1 = math.dist(keepPoints[curFace[1]],keepPoints[curFace[0]])
                s2 = math.dist(keepPoints[curFace[2]],keepPoints[curFace[1]])
                s3 = math.dist(keepPoints[curFace[0]],keepPoints[curFace[2]])
                area += triangleArea(s1,s2,s3)
    obj.close()
    
    #Write newly calculated area to cache
    cache = open(cachePath,'a+')
    cache.write(filename + ' ' + str(area) + '\n')
    cache.close()
    return area


#Find point on aneurysm which is farthest away from the neck plane
def farthestPoint(pathActual,anPoints):
    (A, B, c) = lstsq(np.c_[pathActual[:,0],pathActual[:,1],np.ones(pathActual.shape[0])],pathActual[:,2])[0] #Ax + By + c = z
    return anPoints[np.argmax(np.abs(A*anPoints[:,0]+B*anPoints[:,1]-anPoints[:,2]+c))]


#Convert coordinates of a vertex to indices of a voxel which contains that vertex
def vertexToVoxel(voxelArray, vertex):
    voxelArray = addWhitespace(voxelArray)
    vertex = (int(round(vertex[0]/C_VOXEL_SIZE[0])),int(round(vertex[1]/C_VOXEL_SIZE[1])),int(round(vertex[2]/C_VOXEL_SIZE[2])))
    neighbors = voxelArray[vertex[0]-1:vertex[0]+1,vertex[1]-1:vertex[1]+1,vertex[2]-1:vertex[2]+1]
    offset = np.argwhere(neighbors)[0]
    voxel = np.array([vertex[0]+offset[0]-2,vertex[1]+offset[1]-2,vertex[2]+offset[2]-2])
    return voxel


#Add degenerate vertices to a mesh to display without interpolation
def displayMesh(vertices, faces, normals):
    vertexArray = np.array([vertices[i*3:i*3+3] for i in faces])
    normalArray = np.array([normals[i*3:i*3+3] for i in faces])
    vertexArrayFace = vertexArray.reshape((-1,3,3)) #(face,vertex,axis)
    v1 = vertexArrayFace[:,0,:] - vertexArrayFace[:,1,:]
    v2 = vertexArrayFace[:,0,:] - vertexArrayFace[:,2,:]
    norm = np.cross(v1,v2)
    dot = np.sum(norm*normalArray[::3],axis=1)
    norm[dot<0,:] = norm[dot<0,:]*-1
    norm = norm / np.sqrt((norm**2).sum(-1)).reshape((-1,1))
    norm = np.repeat(norm,3,axis=0)
    return (vertexArray.flatten(),norm.flatten())


#Create a dictionary mapping points in the indexed vertex list to points in the nonindexed list
def getDisplayDictMesh(indexed,display):
    np.set_printoptions(threshold=sys.maxsize)
    vertexList = np.array(list(map(tuple,np.array(display).reshape((-1,3)))),dtype=np.dtype('float,float,float'))
    vertices = np.array(list(map(tuple,np.array(indexed).reshape((-1,3)))),dtype=np.dtype('float,float,float'))
    vertexDict = defaultdict(list)
    for i,v in enumerate(vertices):
        vertexDict[tuple(v)] = list(np.argwhere(vertexList == vertices[i]).flatten())
    return vertexDict


#Convert a list of vertices and faces to a graph
def meshToGraph(vertices, faces):
    graph = nx.Graph()
    vertices = vertices.reshape((-1,3))
    faces = faces.reshape((-1,3))
    np.fromiter((graph.add_node(tuple(v)) for v in vertices),vertices.dtype)
    np.fromiter((graph.add_edges_from([(tuple(vertices[f[0]]),tuple(vertices[f[1]])),(tuple(vertices[f[1]]),tuple(vertices[f[2]])),(tuple(vertices[f[2]]),tuple(vertices[f[0]]))]) for f in faces),vertices.dtype)
    return(graph)


#Identify the set of vertices defined by a boundary line
def selectRegion(graph, boundary, point):
    time1 = time.time()
    boundary = boundary.reshape((-1,3))
    graph = graph.copy()
    graph.remove_nodes_from(list(zip(boundary[:,0],boundary[:,1],boundary[:,2])))
    return np.array(list(map(np.array,nx.shortest_path(graph,tuple(point)).keys())))


#Adds one voxel of whitespace around a 3D array
def addWhitespace(arr):
	arr = np.concatenate((np.zeros((1,arr.shape[1],arr.shape[2]), dtype=arr.dtype), arr, 
			np.zeros((1,arr.shape[1],arr.shape[2]), dtype=arr.dtype)), axis=0)
			
	arr = np.concatenate((np.zeros((arr.shape[0],1,arr.shape[2]), dtype=arr.dtype), arr, 
				np.zeros((arr.shape[0],1,arr.shape[2]), dtype=arr.dtype)), axis=1)
				
	arr = np.concatenate((np.zeros((arr.shape[0],arr.shape[1],1), dtype=arr.dtype), arr, 
				np.zeros((arr.shape[0],arr.shape[1],1), dtype=arr.dtype)), axis=2)
				
	return arr


#Generate a vertex list from a voxel array
def vertexListVoxel(image):
    #Pad image borders so each image voxel has a surrounding 9x9 cube
    image = addWhitespace(image)
    
    #Set image points with no exposed faces to 2
    image[image>0] = 1
    k = np.array([[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]])
    conv = ndimage.convolve(image, k, mode='constant', cval=0.0)
    image[conv==7] = 2
    
    faces = []
    vertices = []
    normals = []
    vertex_index = 0
    vertexDict = defaultdict(list)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            for k in range(1,image.shape[2]-1):
                if image[i,j,k]==1:
                    if image[i-1,j,k]==0:
                        faces.extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertexDict[(i-1)*(image.shape[1]-2)*(image.shape[2]-2) + (j-1)*(image.shape[2]-2) + (k-1)].extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertex_index += 6
                        vertices.extend([i,j,k, i,j+1,k, i,j+1,k+1, i,j,k, i,j+1,k+1, i,j,k+1])
                        normals.extend([-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0])
                    if image[i+1,j,k]==0:
                        faces.extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertexDict[(i-1)*(image.shape[1]-2)*(image.shape[2]-2) + (j-1)*(image.shape[2]-2) + (k-1)].extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertex_index += 6
                        vertices.extend([i+1,j,k, i+1,j+1,k, i+1,j+1,k+1, i+1,j,k, i+1,j+1,k+1, i+1,j,k+1])
                        normals.extend([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])
                    if image[i,j-1,k]==0:
                        faces.extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertexDict[(i-1)*(image.shape[1]-2)*(image.shape[2]-2) + (j-1)*(image.shape[2]-2) + (k-1)].extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertex_index += 6
                        vertices.extend([i,j,k, i+1,j,k, i+1,j,k+1, i,j,k, i+1,j,k+1, i,j,k+1])
                        normals.extend([0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0])
                    if image[i,j+1,k]==0:
                        faces.extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertexDict[(i-1)*(image.shape[1]-2)*(image.shape[2]-2) + (j-1)*(image.shape[2]-2) + (k-1)].extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertex_index += 6
                        vertices.extend([i,j+1,k, i+1,j+1,k, i+1,j+1,k+1, i,j+1,k, i+1,j+1,k+1, i,j+1,k+1])
                        normals.extend([0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0])
                    if image[i,j,k-1]==0:
                        faces.extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertexDict[(i-1)*(image.shape[1]-2)*(image.shape[2]-2) + (j-1)*(image.shape[2]-2) + (k-1)].extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertex_index += 6
                        vertices.extend([i,j,k, i+1,j,k, i+1,j+1,k, i,j,k, i+1,j+1,k, i,j+1,k])
                        normals.extend([0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1])
                    if image[i,j,k+1]==0:
                        faces.extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertexDict[(i-1)*(image.shape[1]-2)*(image.shape[2]-2) + (j-1)*(image.shape[2]-2) + (k-1)].extend([vertex_index,vertex_index+1,vertex_index+2,vertex_index+3,vertex_index+4,vertex_index+5])
                        vertex_index += 6
                        vertices.extend([i,j,k+1, i+1,j,k+1, i+1,j+1,k+1, i,j,k+1, i+1,j+1,k+1, i,j+1,k+1])
                        normals.extend([0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1])
                        
    return faces,vertices,normals,vertexDict


#Add a square face bounded by corners c1-c4 to a vertex list (Helper function for vertexListVoxelNoDuplicates)
def addFaceNoDuplicates(c1,c2,c3,c4, faces, vertices, vertex_indices, vertex_index):
    if vertex_indices[c1]==-1:
        vertex_indices[c1] = vertex_index
        vertex_index += 1
        vertices.extend(list(c1))
    
    if vertex_indices[c2]==-1:
        vertex_indices[c2] = vertex_index
        vertex_index += 1
        vertices.extend(list(c2))
        
    if vertex_indices[c3]==-1:
        vertex_indices[c3] = vertex_index
        vertex_index += 1
        vertices.extend(list(c3))
        
    if vertex_indices[c4]==-1:
        vertex_indices[c4] = vertex_index
        vertex_index += 1
        vertices.extend(list(c4))
        
    faces.extend([vertex_indices[c1],vertex_indices[c2],vertex_indices[c3],vertex_indices[c1],vertex_indices[c3],vertex_indices[c4]])
    
    return vertex_index


#Generate a vertex list from a voxel array without duplicating vertices
def vertexListVoxelNoDuplicates(image):
    #Pad image borders so each image voxel has a surrounding 9x9 cube
    image = addWhitespace(image)
    
    #Set image points with no exposed faces to 2
    image[image>0] = 1
    k = np.array([[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]])
    conv = ndimage.convolve(image, k, mode='constant', cval=0.0)
    image[conv==7] = 2
    
    faces = []
    vertices = []
    vertex_indices = np.ones(image.shape,dtype=np.int64)*-1
    vertex_index = 0
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            for k in range(1,image.shape[2]-1):
                if image[i,j,k]==1:
                    if image[i-1,j,k]==0:
                        vertex_index = addFaceNoDuplicates((i,j,k),(i,j+1,k),(i,j+1,k+1),(i,j,k+1),faces,vertices,vertex_indices,vertex_index)
                    if image[i+1,j,k]==0:
                        vertex_index = addFaceNoDuplicates((i+1,j,k),(i+1,j+1,k),(i+1,j+1,k+1),(i+1,j,k+1),faces,vertices,vertex_indices,vertex_index)
                    if image[i,j-1,k]==0:
                        vertex_index = addFaceNoDuplicates((i,j,k),(i+1,j,k),(i+1,j,k+1),(i,j,k+1),faces,vertices,vertex_indices,vertex_index)
                    if image[i,j+1,k]==0:
                        vertex_index = addFaceNoDuplicates((i,j+1,k),(i+1,j+1,k),(i+1,j+1,k+1),(i,j+1,k+1),faces,vertices,vertex_indices,vertex_index)
                    if image[i,j,k-1]==0:
                        vertex_index = addFaceNoDuplicates((i,j,k),(i+1,j,k),(i+1,j+1,k),(i,j+1,k),faces,vertices,vertex_indices,vertex_index)
                    if image[i,j,k+1]==0:
                        vertex_index = addFaceNoDuplicates((i,j,k+1),(i+1,j,k+1),(i+1,j+1,k+1),(i,j+1,k+1),faces,vertices,vertex_indices,vertex_index)
                        
    return faces,vertices
    

#Calculate Dice Similarity Coefficient or Jaccard Index for calculated mesh segmentation
def similarityMesh(filename, points):
    #Get paths to files
    objPath = C_OBJ_PATH[0] + filename + C_OBJ_PATH[1]
    anPath = C_AN_PATH[0] + filename + C_AN_PATH[1]
    
    #Create list of which points in the object belong to each segmentation
    #0=neither, 1=actual, 2=calculated, 3=both
    an = open(anPath)
    seg = []
    points = list(map(tuple,points.reshape((-1,3))))
    for line in an:
        values = line.split()
        membership = 0
        if len(values)>2 and (float(values[0]),float(values[1]),float(values[2])) in points:
            if values[-1]=='1' or values[-1]=='2':
                membership = 3
            else:
                membership = 2
        elif values[-1]=='1' or values[-1]=='2':
            membership = 1
        seg.append((membership,(float(values[0]),float(values[1]),float(values[2]))))
    an.close()
    
    #Calculate surface areas
    obj = open(objPath)
        
    areaCalc = 0
    areaActual = 0
    areaIntersection = 0
    areaUnion = 0
    for line in obj:
        values = line.split()
        if len(values)>0 and values[0] == 'f':
            curFace = (int(values[1].split('//')[0])-1,int(values[2].split('//')[0])-1,int(values[3].split('//')[0])-1)
            members = (seg[curFace[0]][0],seg[curFace[1]][0],seg[curFace[2]][0])
            points = (seg[curFace[0]][1],seg[curFace[1]][1],seg[curFace[2]][1])
            
            #Current face does not belong to either segmentation
            if 0 in members:
                pass
            
            #Current face belongs to both segmentations
            elif members == (3,3,3):
                curArea = triangleAreaCoords(points[0],points[1],points[2])
                areaCalc += curArea
                areaActual += curArea
                areaIntersection += curArea
                areaUnion += curArea
                
            #Current face belongs only to calculated segmentation
            elif (2 in members) and not(1 in members):
                curArea = triangleAreaCoords(points[0],points[1],points[2])
                areaCalc += curArea
                areaUnion += curArea
            
            #Current face belongs only to actual segmentation
            elif (1 in members) and not (2 in members):
                curArea = triangleAreaCoords(points[0],points[1],points[2])
                areaActual += curArea
                areaUnion += curArea
                
    obj.close()
    
    if areaUnion == 0: return {'DSC':1.0, 'JI':1.0}
    
    DSC = 2*areaIntersection/(areaCalc+areaActual)
    JI = areaIntersection/areaUnion
    return {'DSC':DSC, 'JI':JI}


#Calculate Dice Similarity Coefficient or Jaccard Index for calculated voxel segmentation
def similarityVoxel(obj, calc, connectivity='26'):
    if connectivity == '6':
        actual = obj.an_voxelArray6
    else:
        actual = obj.an_voxelArray26
    
    areaActual = np.count_nonzero(actual)
    areaCalc = np.count_nonzero(calc)
    areaIntersection = np.count_nonzero(np.logical_and(actual,calc))
    areaUnion = np.count_nonzero(np.logical_or(actual,calc))
    
    if areaUnion == 0: return {'DSC':1.0, 'JI':1.0}
    
    DSC = 2*areaIntersection/(areaCalc+areaActual)
    JI = areaIntersection/areaUnion
    return {'DSC':DSC, 'JI':JI}


#Calculate Dice Similarity Coefficient and Jaccard Index
def similarity(obj, calc, mode, connectivity='26'):
    if mode=='mesh':
        return similarityMesh(obj.filename, calc)
    elif mode=='voxel':
        return similarityVoxel(obj, calc, connectivity)

    
#Set a group of vertices to a specified color in an indexed vertex list
def recolorVerticesIndexed(obj, vertices, newColor):
    vertexList = np.array(obj.vertices).reshape((-1,3))
    vertices = np.array(vertices).reshape((-1,3))
    
    indices = np.full(vertexList.shape[0],False)
    for x in vertices:
        indices = np.bitwise_or(indices,(vertexList == x).all(axis=1))
    
    colorList = np.array(obj.colors).reshape((-1,3))
    colorList[indices] = newColor
    obj.colors = tuple(colorList.flatten())
    
    
#Set a group of vertices to a specified color (recolors every face associated with each vertex)
def recolorVerticesOr(obj, vertices, newColor):
    vertexList = np.array(obj.displayVertices).reshape((-1,3))
    vertices = np.array(vertices).reshape((-1,3))
    
    indices = np.full(vertexList.shape[0],False)
    for x in vertices:
        indices = np.logical_or(indices,(vertexList == x).all(axis=1))
    indices = np.repeat(np.logical_or.reduce(indices.reshape((-1,3)),axis=1),3,axis=0)
    
    colorList = np.array(obj.displayColors).reshape((-1,3))
    colorList[indices] = newColor
    obj.displayColors = tuple(colorList.flatten())
    
    
#Set a group of vertices to a specified color (only recolors faces where every vertex has been recolored)
def recolorVerticesAnd(obj, vertices, newColor):     
    recolorVertices = list(map(tuple,np.array(vertices).reshape((-1,3))))
    indices = np.hstack(itemgetter(*recolorVertices)(obj.displayDict)).astype(int)
    
    colorAr = np.full(len(obj.displayVertices),False)
    colorAr[indices] = True
    colorAr = colorAr.reshape((-1,3))
    colorAr = np.all(colorAr,axis=1)
    colorAr = np.repeat(colorAr,3)
    indices = np.argwhere(colorAr).flatten()
    
    arColors = np.array(obj.displayColors)
    arColors[indices*3] = newColor[0]
    arColors[indices*3+1] = newColor[1]
    arColors[indices*3+2] = newColor[2]
    obj.displayColors = tuple(arColors)
    
    
#Set a group of vertices to a specified color (assumes each set of 6 vertices defines a rectangular face)
def recolorVerticesVoxelOr(obj, vertices, newColor):
    vertexList = np.array(obj.displayVertices).reshape((-1,3))
    vertices = np.array(vertices).reshape((-1,3))
    
    indices = np.full(vertexList.shape[0],False)
    for x in vertices:
        indices = np.logical_or(indices,(vertexList == x).all(axis=1))
    indices = np.repeat(np.logical_or.reduce(indices.reshape((-1,6)),axis=1),6,axis=0)
    
    colorList = np.array(obj.displayColors).reshape((-1,3))
    colorList[indices] = newColor
    obj.displayColors = tuple(colorList.flatten())
    
    
#Set a group of vertices to a specified color (assumes each set of 6 vertices defines a rectangular face)
def recolorVerticesVoxelAnd(obj, vertices, newColor):
    vertexList = np.array(obj.displayVertices).reshape((-1,3))
    vertices = np.array(vertices).reshape((-1,3))
    
    indices = np.full(vertexList.shape[0],False)
    for x in vertices:
        indices = np.logical_or(indices,(vertexList == x).all(axis=1))
    indices = np.repeat(np.logical_and.reduce(indices.reshape((-1,6)),axis=1),6,axis=0)
    
    colorList = np.array(obj.displayColors).reshape((-1,3))
    colorList[indices] = newColor
    obj.displayColors = tuple(colorList.flatten())


#Determine if four points form a unit square
def isUnitSquare(points):
    #Check if coplanar
    if not (np.all(points[:,0]==points[:,0][0]) or np.all(points[:,1]==points[:,1][0]) or np.all(points[:,2]==points[:,2][0])): return False
    
    #Check distance apart
    if np.all((points[1]-points[0])==0) or np.all((points[2]-points[1])==0) or np.all((points[3]-points[2])==0): return False
    if np.any((points[1]-points[0])>1.01*np.array(C_VOXEL_SIZE)) or np.any((points[2]-points[1])>1.01*np.array(C_VOXEL_SIZE)) or np.any((points[3]-points[2])>1.01*np.array(C_VOXEL_SIZE)): return False
    
    #Check if connections at 90 degree angles
    return np.any(np.cross((points[1]-points[0])/np.linalg.norm(points[1]-points[0]),(points[2]-points[1])/np.linalg.norm(points[2]-points[1]))>.99) and np.any(np.cross((points[2]-points[1])/np.linalg.norm(points[2]-points[1]),(points[3]-points[2])/np.linalg.norm(points[3]-points[2]))>.99)


#Create a list of voxels which are separated by a boundary 
def createBoundaryList(an_boundary):
    boundaryList = []
    for i in range(an_boundary.shape[0]):
        (p1,p2) = np.concatenate((an_boundary[i:],an_boundary[:i]))[0:2]
        b = np.array([min(p1[0],p2[0]),min(p1[1],p2[1]),min(p1[2],p2[2])])
        (v1,v2) = (np.zeros(3).astype(int),np.zeros(3).astype(int))
        v1[np.argwhere(np.abs(p2-p1)==0)[0]] = 1
        v2[np.argwhere(np.abs(p2-p1)==0)[1]] = 1
        boundaryList.extend([(tuple(b),tuple(b-v1)),(tuple(b),tuple(b-v2)),(tuple(b),tuple(b-v1-v2)), 
                             (tuple(b-v1),tuple(b)),(tuple(b-v1),tuple(b-v2)),(tuple(b-v1),tuple(b-v1-v2)), 
                             (tuple(b-v2),tuple(b)),(tuple(b-v2),tuple(b-v1)),(tuple(b-v2),tuple(b-v1-v2)), 
                             (tuple(b-v1-v2),tuple(b)),(tuple(b-v1-v2),tuple(b-v1)),(tuple(b-v1-v2),tuple(b-v2))])
    return boundaryList


#Create a voxel mask for an annotation in a voxel array based on a list of vertices
def getVoxelAnMask(obj,an_vertices,an_boundary):
    an_vertices = np.array(an_vertices).reshape((-1,3))
    an_vertices = np.column_stack((an_vertices[:,0]/C_VOXEL_SIZE[0],an_vertices[:,1]/C_VOXEL_SIZE[1],an_vertices[:,2]/C_VOXEL_SIZE[2])).astype(int)
    voxelArray = addWhitespace(obj.voxelArray)
    voxelArray[voxelArray>0] = 1
    
    mask = np.zeros(voxelArray.shape)
    for v in an_vertices:
        mask[v[0]-1:v[0]+1,v[1]-1:v[1]+1,v[2]-1:v[2]+1] = voxelArray[v[0]-1:v[0]+1,v[1]-1:v[1]+1,v[2]-1:v[2]+1]
    
    mask_6connected = mask[1:-1,1:-1,1:-1].astype(int)
    
    k = np.array([[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]])
    conv = ndimage.convolve(voxelArray, k, mode='constant', cval=0.0)
    voxelArray[conv==7] = 0
    mask_26connected = np.logical_and(mask_6connected,voxelArray[1:-1,1:-1,1:-1]).astype(int)

    return mask_6connected,mask_26connected
    

#Recolor the faces associated with a set of voxels
def recolorVoxelMask(obj, mask, newColor):
    coordinates = np.argwhere(mask)
    voxels = coordinates[:,0]*mask.shape[1]*mask.shape[2] + coordinates[:,1]*mask.shape[2] + coordinates[:,2]
    indices = np.hstack(itemgetter(*voxels)(obj.displayDict)).astype(int)
    arColors = np.array(obj.displayColors)
    arColors[indices*3] = newColor[0]
    arColors[indices*3+1] = newColor[1]
    arColors[indices*3+2] = newColor[2]
    obj.displayColors = tuple(arColors)
    
    
#Read vertex, face, and annotation data for file
def readDataMesh(filename, force_recalc=C_FORCE_RECALC):
    #Create file paths
    objPath = C_OBJ_PATH[0] + filename + C_OBJ_PATH[1]
    anPath = C_AN_PATH[0] + filename + C_AN_PATH[1]
    
    #Chech for cached version
    cachePath = C_MESH_CACHE_PATH[0] + filename + C_MESH_CACHE_PATH[1]
    if os.path.isfile(cachePath) and os.path.getmtime(cachePath)>os.path.getmtime(objPath) and os.path.getmtime(cachePath)>os.path.getmtime(anPath) and not force_recalc:
        data = pickle.load(open(cachePath,'rb'))
        
        #Color mesh based on segmentation
        try:
            (testVertices,testPath) = pickle.load(open(C_CALC_CACHE_PATH[0] + data.filename + C_CALC_CACHE_PATH[1],'rb'))
            recolorVerticesAnd(data,testVertices,C_SURFACE_CALC_COLOR)
            recolorVerticesAnd(data,np.array(list(set(list(map(tuple,testVertices.reshape((-1,3))))).intersection(list(map(tuple,np.append(data.an_vertices,data.an_boundary).reshape((-1,3))))))).flatten(),C_SURFACE_OVERLAP_COLOR)
            data.calc_boundary = testPath
        except:
            pass
        
        data.objVertexList = pyglet.graphics.vertex_list(data.displayNumVertices,('v3f',data.displayVertices),('n3f',data.displayNormals),('c3B',data.displayColors))
        data.anVertexList = pyglet.graphics.vertex_list(data.numAnPoints,('v3f', data.an_boundary),('c3B',C_PATH_ACTUAL_COLOR*data.numAnPoints))
        return data
    
    #Read faces and vertices from obj file
    obj = open(objPath)
    vertices = []
    faces = []
    for line in obj:
        values = line.split()
        if len(values)>0 and values[0] == 'v':
            vertices.extend([float(values[1]),float(values[2]),float(values[3])])
        if len(values)>0 and values[0] == 'f':
            faces.extend([int(values[1].split('//')[0])-1,int(values[2].split('//')[0])-1,int(values[3].split('//')[0])-1])
    obj.close()
    
    #Read annotation data from an file
    an = open(anPath)
    normals = []
    an_boundary = []
    an_vertices = []
    for line in an:
        values = line.split()
        if len(values)>0:
            normals.extend([float(values[3]),float(values[4]),float(values[5])])
        if len(values)>0 and values[-1]=='2':
            an_boundary.extend([float(values[0]),float(values[1]),float(values[2])])
        if len(values)>0 and values[-1]=='1':
            an_vertices.extend([float(values[0]),float(values[1]),float(values[2])])
    an.close()
    an_boundary = sortPath(an_boundary)
    
    data = AnDataMesh(filename,vertices,faces,normals,an_boundary,an_vertices)
    pickle.dump(data,open(cachePath,'wb'))
    
    #Color mesh based on segmentation
    try:
        (testVertices,testPath) = pickle.load(open(C_CALC_CACHE_PATH[0] + data.filename + C_CALC_CACHE_PATH[1],'rb'))
        recolorVerticesAnd(data,testVertices,C_SURFACE_CALC_COLOR)
        recolorVerticesAnd(data,np.array(list(set(list(map(tuple,testVertices.reshape((-1,3))))).intersection(list(map(tuple,np.append(data.an_vertices,data.an_boundary).reshape((-1,3))))))).flatten(),C_SURFACE_OVERLAP_COLOR)
        data.calc_boundary = testPath
    except:
        pass
    
    data.objVertexList = pyglet.graphics.vertex_list(data.displayNumVertices,('v3f',data.displayVertices),('n3f',data.displayNormals),('c3B',data.displayColors))
    data.anVertexList = pyglet.graphics.vertex_list(data.numAnPoints,('v3f', data.an_boundary),('c3B',C_PATH_ACTUAL_COLOR*data.numAnPoints))
    
    return data


#Read an array of voxels and convert it to a displayable object
def readDataVoxel(filename, force_recalc=C_FORCE_RECALC):
    objPath = C_VOXEL_PATH[0] + filename + C_VOXEL_PATH[1]
    anPath = C_VOXEL_AN_PATH[0] + filename + C_VOXEL_AN_PATH[1]
    
    #Chech for cached version
    cachePath = C_VOXEL_CACHE_PATH[0] + filename + C_VOXEL_CACHE_PATH[1]
    if os.path.isfile(cachePath) and os.path.getmtime(cachePath)>os.path.getmtime(objPath) and os.path.getmtime(cachePath)>os.path.getmtime(anPath) and not force_recalc:   
        data = pickle.load(open(cachePath,'rb'))
        
        #Color voxels based on segmentation
        try:
            (testArray,testPath) = pickle.load(open(C_CALC_CACHE_PATH[0] + data.filename + C_CALC_CACHE_PATH[1],'rb'))
            recolorVoxelMask(data,data.an_voxelArray26,C_SURFACE_ACTUAL_COLOR)
            recolorVoxelMask(data,testArray,C_SURFACE_CALC_COLOR)
            recolorVoxelMask(data,np.logical_and(testArray,data.an_voxelArray26),C_SURFACE_OVERLAP_COLOR)
            data.calc_boundary = testPath
        except:
            pass
            
        data.objVertexList = pyglet.graphics.vertex_list(data.displayNumVertices,('v3f',data.displayVertices),('n3f',data.displayNormals),('c3B',data.displayColors))
        data.anVertexList = pyglet.graphics.vertex_list(data.numAnPoints,('v3f', data.an_boundary),('c3B',C_PATH_ACTUAL_COLOR*data.numAnPoints))
        return data
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = pydicom.dcmread(objPath).pixel_array
    (faces,vertices) = vertexListVoxelNoDuplicates(image)
    vertices[0::3] = [v*C_VOXEL_SIZE[0] for v in vertices[0::3]]
    vertices[1::3] = [v*C_VOXEL_SIZE[1] for v in vertices[1::3]]
    vertices[2::3] = [v*C_VOXEL_SIZE[2] for v in vertices[2::3]]
    
    (displayFaces,displayVertices,displayNormals,displayDict) = vertexListVoxel(image)
    displayVertices[0::3] = [v*C_VOXEL_SIZE[0] for v in displayVertices[0::3]]
    displayVertices[1::3] = [v*C_VOXEL_SIZE[1] for v in displayVertices[1::3]]
    displayVertices[2::3] = [v*C_VOXEL_SIZE[2] for v in displayVertices[2::3]]
    
    anFile = open(anPath,'r')
    an_vertices = []
    an_boundary = [float(x) for x in anFile.readline().split()]
    for line in anFile:
        values = line.split()
        if len(values)>0 and values[-1]=='1':
            an_vertices.extend([float(values[0]),float(values[1]),float(values[2])])
    anFile.close()
    
    data = AnDataVoxel(filename,vertices,faces,an_boundary,an_vertices,displayVertices,displayFaces,displayNormals,displayDict,image)
    pickle.dump(data,open(cachePath,'wb'))
    
    #Color voxels based on segmentation
    try:
        (testArray,testPath) = pickle.load(open(C_CALC_CACHE_PATH[0] + data.filename + C_CALC_CACHE_PATH[1],'rb'))
        recolorVoxelMask(data,data.an_voxelArray26,C_SURFACE_ACTUAL_COLOR)
        recolorVoxelMask(data,testArray,C_SURFACE_CALC_COLOR)
        recolorVoxelMask(data,np.logical_and(testArray,data.an_voxelArray26),C_SURFACE_OVERLAP_COLOR)
        data.calc_boundary = testPath
        
    except:
        pass
    
    data.objVertexList = pyglet.graphics.vertex_list(data.displayNumVertices,('v3f',data.displayVertices),('n3f',data.displayNormals),('c3B',data.displayColors))
    data.anVertexList = pyglet.graphics.vertex_list(data.numAnPoints,('v3f', data.an_boundary),('c3B',C_PATH_ACTUAL_COLOR*data.numAnPoints))
    
    return data


#Read data in the specified format
def readData(filename,mode):
    if mode=='mesh':
        return readDataMesh(filename)
    if mode=='voxel':
        return readDataVoxel(filename)


#Divides the annotation data of a file based on a plane
def splitFile(nameIn, nameOut1, nameOut2, plane):
    inFile = open(C_AN_PATH[0] + nameIn + C_AN_PATH[1])
    outFile1 = open(C_AN_PATH[0] + nameOut1 + C_AN_PATH[1], 'w+')
    outFile2 = open(C_AN_PATH[0] + nameOut2 + C_AN_PATH[1], 'w+')
    
    v1 = np.array(plane[0])-np.array(plane[1])
    v2 = np.array(plane[0])-np.array(plane[2])
    normal = np.cross(v1,v2)
    p = np.array(plane[0])
    
    for line in inFile:
        values = line.split()
        if len(values)>0 and np.dot((np.array([float(values[0]),float(values[1]),float(values[2])])-p),normal)>0:
            outFile1.write(line)
            outFile2.write(values[0]+' '+values[1]+' '+values[2]+' '+values[3]+' '+values[4]+' '+values[5]+' '+'0\n')
        elif len(values)>0:
            outFile2.write(line)
            outFile1.write(values[0]+' '+values[1]+' '+values[2]+' '+values[3]+' '+values[4]+' '+values[5]+' '+'0\n')
        else:
            outFile1.write(line)
            outFile2.write(line)
    inFile.close()
    outFile1.close()
    outFile2.close()


#Create formatted text string from object attributes
def descString(name,obj):
    return '{font_size ' + str(C_GUI_TEXT_TITLE_SIZE) + '}{color '+ str(C_GUI_TEXT_COLOR) + '}{align "right"}' + name + '\n\n' + \
           '{font_size ' + str(C_GUI_TEXT_SIZE) + '}' + str(obj.numVertices) + ' Vertices, ' + str(len(obj.faces)) + ' Faces' + '\n\n' + \
           'Segment Dimensions: ' + str(round(obj.vesselDim.x)) + ' x ' + str(round(obj.vesselDim.y)) + ' x ' + str(round(obj.vesselDim.z)) + ' mm' + '\n\n' + \
           'Aneurysm Dimensions: ' + str(round(obj.anDim.x)) + ' x ' + str(round(obj.anDim.y)) + ' x ' + str(round(obj.anDim.z)) + ' mm'


#Writes a 3D array of voxels to a DICOM file
def printDICOM(arr,ds,filename):
	im = arr.copy(order='C')
	ds.PixelData = im
	pydicom.dcmwrite(filename, ds, True)


#Check that all arguments passed to function are valid
def sanitizeArguments(arguments,mode):
    arguments = arguments.split(',')
    valid_voxel = ['filename','point','voxel_array']
    valid_mesh = ['filename','point','vertices','faces']
    if mode=='voxel':        
        for arg in arguments:
            if arg not in valid_voxel:
                if arg in valid_mesh:
                    raise Exception('Argument \'' + arg + '\' is not available in ' + mode + ' mode')
                else:
                    raise Exception('Unknown argument: \'' + arg + '\'')
                
    elif mode=='mesh':               
        for arg in arguments:
            if arg not in valid_mesh:
                if arg in valid_voxel:
                    raise Exception('Argument \'' + arg + '\' is not available in ' + mode + ' mode')
                else:
                    raise Exception('Unknown argument: \'' + arg + '\'')


#Create an aneurysm boundary from a voxel segmentation
def voxelArrayToPath(seg,array,obj):
    #Pad inputs and create local copies to modify
    seg = addWhitespace(seg)
    array = addWhitespace(array)
    segL = np.copy(seg)
    arrayL = np.copy(array)
  
    #Set voxels with no exposed faces to 0
    arrayL[arrayL>0] = 1
    k = np.array([[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]])
    conv = ndimage.convolve(arrayL, k, mode='constant', cval=0)
    arrayL[conv==7] = 0
    segL[arrayL==0] = 0
    
    #Remove segmentation points from image array
    arrayL[segL>0] = 0
    
    #If the segmentation includes the whole image, return no border
    if not arrayL.any():
        return []
    
    #Convert segmentation and image boundary voxel arrays to vertex arrays
    seg_vert = np.zeros((segL.shape[0]+1,segL.shape[1]+1,segL.shape[2]+1))
    seg_vert[:-1,:-1,:-1] = segL
    seg_vert[1:,:-1,:-1] = np.logical_or(seg_vert[1:,:-1,:-1],seg_vert[:-1,:-1,:-1])
    seg_vert[:,1:,:-1] = np.logical_or(seg_vert[:,1:,:-1],seg_vert[:,:-1,:-1])
    seg_vert[:,:,1:] = np.logical_or(seg_vert[:,:,1:],seg_vert[:,:,:-1])
    
    array_vert = np.zeros((arrayL.shape[0]+1,arrayL.shape[1]+1,arrayL.shape[2]+1))
    array_vert[:-1,:-1,:-1] = arrayL
    array_vert[1:,:-1,:-1] = np.logical_or(array_vert[1:,:-1,:-1],array_vert[:-1,:-1,:-1])
    array_vert[:,1:,:-1] = np.logical_or(array_vert[:,1:,:-1],array_vert[:,:-1,:-1])
    array_vert[:,:,1:] = np.logical_or(array_vert[:,:,1:],array_vert[:,:,:-1])
    
    #Remove internal vertices
    internal_vert = np.zeros((array.shape[0]+1,array.shape[1]+1,array.shape[2]+1))
    internal_vert[:-1,:-1,:-1] = array
    internal_vert[:-1,:-1,:-1][internal_vert[:-1,:-1,:-1]>0] = 2
    internal_vert[:-1,:-1,:-1][internal_vert[:-1,:-1,:-1]==0] = 1
    internal_vert[:-1,:-1,:-1][internal_vert[:-1,:-1,:-1]==2] = 0
    internal_vert[1:,:-1,:-1] = np.logical_or(internal_vert[1:,:-1,:-1],internal_vert[:-1,:-1,:-1])
    internal_vert[:,1:,:-1] = np.logical_or(internal_vert[:,1:,:-1],internal_vert[:,:-1,:-1])
    internal_vert[:,:,1:] = np.logical_or(internal_vert[:,:,1:],internal_vert[:,:,:-1]) 
 
    #Identify vertices which are included in a face in the segmented image, original image, and an empty voxel
    boundary_vertices = np.logical_and(np.logical_and(seg_vert,array_vert),internal_vert)
    
    #Convert to list of coordinates
    boundary_vertices = list(np.argwhere(boundary_vertices))
    boundary_vertices = [tuple(v) for v in boundary_vertices]
    
    #Remove extraneous vertices caused by exposed corners
    boundary_vertices_keep = []
    for v in boundary_vertices:
        sv = segL[v[0]-1:v[0]+1,v[1]-1:v[1]+1,v[2]-1:v[2]+1]
        av = np.logical_not(array[v[0]-1:v[0]+1,v[1]-1:v[1]+1,v[2]-1:v[2]+1])        
        if (sv[0,0,0] and (av[1,0,0] or av[0,1,0] or av[0,0,1])) or \
           (sv[0,0,1] and (av[1,0,1] or av[0,1,1] or av[0,0,0])) or \
           (sv[0,1,0] and (av[1,1,0] or av[0,1,1] or av[0,0,0])) or \
           (sv[0,1,1] and (av[1,1,1] or av[0,0,1] or av[0,1,0])) or \
           (sv[1,0,0] and (av[0,0,0] or av[1,1,0] or av[1,0,1])) or \
           (sv[1,0,1] and (av[0,0,1] or av[1,1,1] or av[1,0,0])) or \
           (sv[1,1,0] and (av[0,1,0] or av[1,1,1] or av[1,0,0])) or \
           (sv[1,1,1] and (av[0,1,1] or av[1,0,1] or av[1,1,0])):
            
            sv = np.logical_not(array[v[0]-1:v[0]+1,v[1]-1:v[1]+1,v[2]-1:v[2]+1])
            av = arrayL[v[0]-1:v[0]+1,v[1]-1:v[1]+1,v[2]-1:v[2]+1]
            if (av[0,0,0] and (sv[1,0,0] or sv[0,1,0] or sv[0,0,1])) or \
               (av[0,0,1] and (sv[1,0,1] or sv[0,1,1] or sv[0,0,0])) or \
               (av[0,1,0] and (sv[1,1,0] or sv[0,1,1] or sv[0,0,0])) or \
               (av[0,1,1] and (sv[1,1,1] or sv[0,0,1] or sv[0,1,0])) or \
               (av[1,0,0] and (sv[0,0,0] or sv[1,1,0] or sv[1,0,1])) or \
               (av[1,0,1] and (sv[0,0,1] or sv[1,1,1] or sv[1,0,0])) or \
               (av[1,1,0] and (sv[0,1,0] or sv[1,1,1] or sv[1,0,0])) or \
               (av[1,1,1] and (sv[0,1,1] or sv[1,0,1] or sv[1,1,0])):
           
                boundary_vertices_keep.append(v)
                
    boundary_vertices = boundary_vertices_keep
    boundary = [(p[0]*C_VOXEL_SIZE[0],p[1]*C_VOXEL_SIZE[1],p[2]*C_VOXEL_SIZE[2]) for p in boundary_vertices]
    
    graph = obj.graph.copy()
    graph.remove_nodes_from(graph.nodes()-boundary)
    for x,y in graph.edges:
        if (x[0]!=y[0] and x[1]!=y[1]) or (x[0]!=y[0] and x[2]!=y[2]) or (x[1]!=y[1] and x[2]!=y[2]): graph.remove_edge(x,y)
    cycles = sorted(nx.simple_cycles(graph.to_directed()),key=len)

    if len(cycles)==0 or len(cycles[-1])!=len(boundary):
        return []
   
    return list(sum(cycles[-1], ()))


#Create a boundary from a list of segmented vertices
def vertexListToPath(segVertices,obj):
    if len(segVertices)==0: return []
    vertices = list(map(tuple,obj.vertices.reshape((-1,3))))
    segVertices = list(map(tuple,segVertices.reshape((-1,3))))
      
    graph = obj.graph.copy()
    graph.remove_nodes_from(segVertices)
    outside = graph.nodes()
    
    boundary = list(nx.node_boundary(obj.graph,outside))
    
    graph = obj.graph.copy()
    graph.remove_nodes_from(graph.nodes()-boundary)
    cycles = sorted(nx.simple_cycles(graph.to_directed()),key=len)
    
    if len(cycles)==0 or len(cycles[-1])!=len(boundary):
        return []
   
    return list(sum(cycles[-1], ()))


# first argument: file path to test function
# second argument: prototype of test function in format function_name(arg1,arg2,...)
#   - voxel mode:
#       . available arguments: filename (string), point (1D numpy array of 3 ints), voxel_array (3D numpy array of ints)
#       . return type: segmented_array (3D numpy array of ints with values >0 indicating a voxel belongs to the segmentation)
#   - mesh mode:
#       . available arguments: filename(string), point (1D numpy array of 3 floats), vertices (Nx3 numpy array of floats), faces (Mx3 numpy array of ints)
#       . return type: segmented_vertices (Yx3 numpy array of floats indicating vertices which belong to the segmentation)
# third argument: test mode (mesh or voxel)
# -i/--include: filenames to use for testing
# -x/--exclude: filenames to exclude from testing (ignored if -i is also used)
# -o/--output: location to store output file (defaults to output/out_[timestamp].txt in directory of test script)
# -r/--render: displays rendering of results (default false)
def main():
    #Variables for interfacing with handlers
    fileList = []
    fileIndex = 0
    window = None
    obj = None
    rotation_matrix = np.identity(4,dtype=ctypes.c_float)
    pan_offset = np.zeros(3)
    zoom_scale = 1
    overlay = None
    description = None
    mode = 'voxel'
  
  
    #Event handlers
    
    #Adjust view boundaries when window is resized
    def on_resize(width, height):
        #Adjust GUI position
        rightButton.position = (window.width-imRightUnpressed.width-C_GUI_BORDER_OFFSET,C_GUI_BORDER_OFFSET)
        rightLabel.position = (window.width-imRightUnpressed.width/2-C_GUI_BORDER_OFFSET,C_GUI_BORDER_OFFSET+rightButton.height)
        leftButton.position = (window.width-imLeftUnpressed.width-imRightUnpressed.width-C_GUI_BORDER_OFFSET-C_GUI_ARROW_SPACING,C_GUI_BORDER_OFFSET)
        leftLabel.position = (window.width-imRightUnpressed.width-imLeftUnpressed.width/2-C_GUI_BORDER_OFFSET-C_GUI_ARROW_SPACING,C_GUI_BORDER_OFFSET+rightButton.height)
        description.position = (window.width-description.width-C_GUI_BORDER_OFFSET ,window.height-C_GUI_BORDER_OFFSET)
        textLabel.position = (leftButton.x,C_GUI_BORDER_OFFSET+imRightUnpressed.height+rightLabel.content_height+C_GUI_ELEMENT_SPACING)
        textSelect.position = (leftButton.x+C_GUI_TEXT_SPACING+textLabel.content_width,C_GUI_BORDER_OFFSET+imRightUnpressed.height+rightLabel.content_height+C_GUI_ELEMENT_SPACING)


    #Zoom when mouse wheel is rotated
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        nonlocal zoom_scale
        
        zoom_scale = zoom_scale * (1+C_ZOOM_STEP*scroll_y)
        glLineWidth(min(C_LINE_WIDTH*zoom_scale,C_LINE_WIDTH))
        glPointSize(min(C_POINT_SIZE*zoom_scale,C_POINT_SIZE))
            
            
    #Pan or rotate when mouse drag occurs       
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        nonlocal rotation_matrix, pan_offset
        
        #Rotate on left mouse drag
        if buttons & mouse.LEFT:
            axis = [dy/math.sqrt(dx**2+dy**2), -dx/math.sqrt(dx**2+dy**2), 0]
            axis = axis/np.linalg.norm(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            angle = C_ROTATION_STEP * math.sqrt(dx**2+dy**2)
            cos = math.cos(angle)
            sin = math.sin(angle)
            rotation_matrix = np.matmul(rotation_matrix,np.array([[cos+x*x*(1-cos),x*y*(1-cos)-z*sin,x*z*(1-cos)+y*sin,0],
                                                                  [y*x*(1-cos)+z*sin,cos+y*y*(1-cos),y*z*(1-cos)-x*sin,0],
                                                                  [z*x*(1-cos)-y*sin,z*y*(1-cos)+x*sin,cos+z*z*(1-cos),0],
                                                                  [0,0,0,1]]))
            
        #Pan on right mouse button drag
        if buttons & mouse.RIGHT:
            w = window.width
            h = window.height
            dist = 2*math.tan(C_VIEW_FOV/2*(math.pi/180))*C_VIEW_START_DIST*obj.radius
            pan_offset += np.array([dx/h*dist,dy/h*dist,0])
            

    #Render object with annotations
    def on_draw():
        
        #Set projection
        window.clear()
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity(); 
        gluPerspective(C_VIEW_FOV,window.width/window.height,0.1,(C_VIEW_START_DIST+zoom_scale)*obj.radius)
        glTranslatef(-obj.center.x,-obj.center.y,-obj.center.z-C_VIEW_START_DIST*obj.radius)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        #Apply pan
        glTranslatef(pan_offset[0],pan_offset[1],pan_offset[2])
        
        #Apply rotation
        glTranslatef(obj.center.x,obj.center.y,obj.center.z)
        glRotationMatrix = np.asfortranarray(np.ravel(rotation_matrix),dtype=ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        glMultMatrixf(glRotationMatrix)
        glTranslatef(-obj.center.x,-obj.center.y,-obj.center.z)
        
        #Apply zoom
        glTranslatef(obj.center.x,obj.center.y,obj.center.z)
        glScalef(zoom_scale,zoom_scale,zoom_scale)
        glTranslatef(-obj.center.x,-obj.center.y,-obj.center.z)
            
        #Draw object
        obj.objVertexList.draw(pyglet.gl.GL_TRIANGLES)
        
        #Draw annotation
        #Add small offset towards camera to prevent z-fighting
        offset = rotation_matrix.dot([0, 0, 1, 0])
        offset = min(C_AN_OFFSET*obj.radius/(zoom_scale*zoom_scale),obj.radius/100)*offset
        
        glTranslatef(offset[0],offset[1],offset[2])
        obj.anVertexList.draw(pyglet.gl.GL_LINE_LOOP)
        pyglet.graphics.draw(int(len(obj.calc_boundary)/3),pyglet.gl.GL_LINE_LOOP,('v3f', obj.calc_boundary),('c3B',C_PATH_CALC_COLOR*int(len(obj.calc_boundary)/3)))
        glTranslatef(-offset[0],-offset[1],-offset[2])
        
        #Draw GUI overlay
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0,window.width,0,window.height)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        overlay.draw()
        
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)


    #Switch screen on left button press
    def on_left_press():
        nonlocal obj,fileIndex
        if fileIndex > 0:
            fileIndex = fileIndex-1
            obj = readData(fileList[fileIndex],mode)
            document = pyglet.text.decode_attributed(descString(fileList[fileIndex],obj))
            description.document = document
            resetView()
        
        
    #Switch screen on right button press
    def on_right_press():
        nonlocal obj,fileIndex,testPath
        if fileIndex < len(fileList)-1:
            fileIndex = fileIndex+1
            obj = readData(fileList[fileIndex],mode)
            document = pyglet.text.decode_attributed(descString(fileList[fileIndex],obj))
            description.document = document
            resetView()

            
    #Jump to object on text entry
    def on_text_select(text):
        nonlocal obj,fileIndex,testPath
        if text in fileList:
            fileIndex = fileList.index(text)
            obj = readData(fileList[fileIndex],mode)
            document = pyglet.text.decode_attributed(descString(fileList[fileIndex],obj))
            description.document = document
            resetView()
        else:
            textSelect.value = '\'' + text + '\'' +  ' not found'
        
        
    #Sort filenames by numerical order
    def fileNameSort(name):
        return int(name.split('AN')[1].split('-')[0])
    
    
    #Reset object manipulation parameters
    def resetView():
        nonlocal rotation_matrix,pan_offset,zoom_scale
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        rotation_matrix = np.identity(4,dtype=ctypes.c_float)
        pan_offset = np.zeros(3)
        zoom_scale = 1
        glLineWidth(min(C_LINE_WIDTH*zoom_scale,C_LINE_WIDTH))
        glPointSize(min(C_POINT_SIZE*zoom_scale,C_POINT_SIZE))
        glLightfv(GL_LIGHT0, GL_POSITION, lightfv(obj.center.x + 1*obj.radius, obj.center.y+1*obj.radius, obj.center.z+1*obj.radius, 1.0))
           

    #Parse arguments
    class Args:
        pass
    argsC = Args()
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('prototype')
    parser.add_argument('mode',choices=['mesh','voxel'])
    parser.add_argument('-i','--include',nargs='+',action='extend',default=[])
    parser.add_argument('-x','--exclude',nargs='+',action='extend',default=[])
    parser.add_argument('-o','--output',default=C_OUTPUT_FILE_DEFAULT)
    parser.add_argument('-r','--render',action='store_true')
    args = parser.parse_args(namespace = argsC)
    mode = argsC.mode   
    
    #Create list of available files
    for f in os.listdir(C_OBJ_PATH[0]):
        if f.endswith('.obj') and os.path.isfile(C_AN_PATH[0] + f.split('_')[0] + C_AN_PATH[1]):
            fileList.append(f.split('_')[0])
    fileList.sort(key=fileNameSort)
    
    if len(argsC.include)>0:
        fileList = [file for file in argsC.include if file in fileList]
        if len(fileList)<len(argsC.include):
            print('file(s) not found:', ', '.join([file for file in argsC.include if file not in fileList]))
    
    elif len(argsC.exclude)>0:
        fileList = [file for file in fileList if file not in argsC.exclude]
        
    if len(fileList) == 0:
        print('No test data selected')
        return
        
    #Parse prototype
    fnName = argsC.prototype.split('(')[0]
    fnArgs = argsC.prototype.split('(')[1].split(')')[0]
    sanitizeArguments(fnArgs,mode)
    
    sys.path.append('/'.join(argsC.path.split('/')[:-1]))
    testMod = importlib.import_module(argsC.path.split('/')[-1].split('.')[0], package=None)
    testFn = getattr(testMod,fnName)
    
    similarityDict = {}
    outFile = open(argsC.output,'w')
    outFile.write('Filename,DSC,JI\n')
    for i,f in enumerate(fileList):
        print(('Running file '+str(i+1)+' of '+str(len(fileList))+': '+f).ljust(shutil.get_terminal_size()[0]-1),end='\r')
        obj = readData(f,mode)
        
        #Prepare input variables
        filename = os.path.abspath(C_VOXEL_PATH[0] + f + C_VOXEL_PATH[1]) if mode=='voxel' else os.path.abspath(C_OBJ_PATH[0] + f + C_OBJ_PATH[1])
        point = obj.an_start_point
        voxel_array = obj.voxelArray if mode=='voxel' else None
        vertices = obj.vertices.reshape((-1,3)) if mode=='mesh' else None
        faces = obj.faces.reshape((-1,3)) if mode=='mesh' else None
        
        #Execute current file
        testResult = eval('testFn(' + fnArgs + ')')
        
        #Process output
        if mode=='voxel':
            testArray = testResult
            testPath = voxelArrayToPath(testArray,obj.voxelArray,obj)
            testResult = (testArray,testPath)
        
        elif mode=='mesh':
            testVertices = testResult
            testPath = vertexListToPath(testVertices,obj)
            testResult = (testVertices,testPath)
        
        pickle.dump(testResult,open(C_CALC_CACHE_PATH[0]+f+C_CALC_CACHE_PATH[1],'wb'))
        
        sim = similarity(obj,testResult[0],mode)
        similarityDict[f] = sim
        outFile.write(f+','+str(round(sim['DSC'],5))+','+str(round(sim['JI'],5))+'\n')
    outFile.close()
    print(('Completed '+str(len(fileList))+' files').ljust(shutil.get_terminal_size()[0]-1))
    
    #End if no rendering
    if not argsC.render: return

    #Load first object
    obj = readData(fileList[fileIndex],mode)
    
    #Initialize window
    window = pyglet.window.Window(resizable=True,visible=False)
    window.set_minimum_size(C_MIN_WINDOW_SIZE_X, C_MIN_WINDOW_SIZE_Y)
    window.maximize()

    #Initialize openGL draw settings
    glLineWidth(C_LINE_WIDTH)
    glPointSize(C_POINT_SIZE)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    #Initialize openGL lighting settings
    lightfv = ctypes.c_float * 4
    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER,GL_TRUE)
    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(obj.center.x + 1*obj.radius, obj.center.y+1*obj.radius, obj.center.z+1*obj.radius, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightfv(0.5, 0.5, 0.5, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightfv(0.0, 0.0, 0.0, 0.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)
    
    #Create overlay
    overlay = pyglet.graphics.Batch()
    
    #Load button images/
    imRightPressed = pyglet.image.load(C_GUI_PATH+'rightArrowPress.png')
    imRightUnpressed = pyglet.image.load(C_GUI_PATH+'rightArrow.png')
    imRightHover = pyglet.image.load(C_GUI_PATH+'rightArrowHover.png')
    imLeftPressed = pyglet.image.load(C_GUI_PATH+'leftArrowPress.png')
    imLeftUnpressed = pyglet.image.load(C_GUI_PATH+'leftArrow.png')
    imLeftHover = pyglet.image.load(C_GUI_PATH+'leftArrowHover.png')
    
    #Add buttons
    rightButton = pyglet.gui.PushButton(window.width-imRightUnpressed.width-C_GUI_BORDER_OFFSET,C_GUI_BORDER_OFFSET,
        imRightPressed,imRightUnpressed,hover=imRightHover,batch=overlay)
    rightLabel = pyglet.text.Label('Next', anchor_x='center', anchor_y='bottom', batch=overlay,
        x=window.width-imRightUnpressed.width/2-C_GUI_BORDER_OFFSET, y=C_GUI_BORDER_OFFSET+rightButton.height)
    rightButton.push_handlers(on_press=on_right_press)
    leftButton = pyglet.gui.PushButton(window.width-imRightUnpressed.width-imLeftUnpressed.width-C_GUI_BORDER_OFFSET-C_GUI_ARROW_SPACING,C_GUI_BORDER_OFFSET,
        imLeftPressed,imLeftUnpressed,hover=imLeftHover,batch=overlay)
    leftLabel = pyglet.text.Label('Previous', anchor_x='center', anchor_y='bottom', batch=overlay,
        x=window.width-imRightUnpressed.width-imLeftUnpressed.width/2-C_GUI_BORDER_OFFSET-C_GUI_ARROW_SPACING, y=C_GUI_BORDER_OFFSET+rightButton.height)
    leftButton.push_handlers(on_press=on_left_press)        
    
    #Add text
    document = pyglet.text.decode_attributed(descString(fileList[fileIndex],obj))
    description = pyglet.text.layout.TextLayout(document, window.width, window.height,
                                                multiline=True,wrap_lines=False,
                                                batch=overlay)
    description.anchor_x = 'left'
    description.anchor_y = 'top'
    description.position = (window.width-description.width-C_GUI_BORDER_OFFSET ,window.height-C_GUI_BORDER_OFFSET)
    
    #Add text entry
    textLabel = pyglet.text.Label('Jump to: ', anchor_x='left', anchor_y='bottom', batch=overlay,
                                  x=leftButton.x, y=C_GUI_BORDER_OFFSET+imRightUnpressed.height+rightLabel.content_height+C_GUI_ELEMENT_SPACING)
    textSelect = pyglet.gui.TextEntry('',
                                      leftButton.x+C_GUI_TEXT_SPACING+textLabel.content_width,
                                      C_GUI_BORDER_OFFSET+imRightUnpressed.height+rightLabel.content_height+C_GUI_ELEMENT_SPACING,
                                      leftButton.width+C_GUI_ARROW_SPACING+rightButton.width - (C_GUI_TEXT_SPACING+textLabel.content_width),
                                      batch=overlay)
    textSelect.set_handler('on_commit',on_text_select)
    
    #Register event handlers
    window.push_handlers(on_resize,on_mouse_scroll,on_mouse_drag,on_draw)
    window.push_handlers(leftButton)
    window.push_handlers(rightButton)
    window.push_handlers(textSelect)
    window.set_visible()

    pyglet.app.run()


if __name__ == '__main__':
    main()
