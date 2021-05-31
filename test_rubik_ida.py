#ai+rubrick ok
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors
import random
from copy import deepcopy
import json
import time
import csv
import pickle

filename = 'RandomForestRegressor18_model.sav'
model = pickle.load(open(filename, 'rb'))
sol_number = 1
plt.style.use('ggplot')

array = np.array([
    [[0, 0, 2], [1, 0, 2], [2, 0, 2]],
    [[0, 0, 1], [1, 0, 1], [2, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
    [[0, 0, 2], [0, 1, 2], [0, 2, 2]],
    [[0, 0, 1], [0, 1, 1], [0, 2, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
    [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
    [[0, 1, 0], [1, 1, 0], [2, 1, 0]],
    [[0, 2, 0], [1, 2, 0], [2, 2, 0]],
    [[2, 0, 0], [2, 0, 1], [2, 0, 2]],
    [[2, 1, 0], [2, 1, 1], [2, 1, 2]],
    [[2, 2, 0], [2, 2, 1], [2, 2, 2]],
    [[2, 0, 2], [1, 0, 2], [0, 0, 2]],
    [[2, 1, 2], [1, 1, 2], [0, 1, 2]],
    [[2, 2, 2], [1, 2, 2], [0, 2, 2]],
    [[0, 2, 0], [1, 2, 0], [2, 2, 0]],
    [[0, 2, 1], [1, 2, 1], [2, 2, 1]],
    [[0, 2, 2], [1, 2, 2], [2, 2, 2]],
])


def manhattan_distance(cube, i, z, corner):
    c1 = array[i, z]
    center = None
    for c in [1, 4, 7, 10, 13, 16]:
        if cube[i, z] == cube[c, 1]:
            center = c
            break

    if corner:
        c2 = array[center - 1, 0]
        d1 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        c2 = array[center - 1, 2]
        d2 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        c2 = array[center + 1, 0]
        d3 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        c2 = array[center + 1, 2]
        d4 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        return min(d1, d2, d3, d4)
    else:
        c2 = array[center - 1, 1]
        d1 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        c2 = array[center, 0]
        d2 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        c2 = array[center, 2]
        d3 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        c2 = array[center + 1, 1]
        d4 = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
        return min(d1, d2, d3, d4)

def corner_edge_sum_max(cube):
    corners = 0
    edges = 0
    for i in range(18):
        if i % 3 == 0 or i % 3 == 2:
            corners = corners + manhattan_distance(cube, i, 0, True) + manhattan_distance(cube, i, 2, True)
            edges = edges + manhattan_distance(cube, i, 1, False)
        else:
            edges = edges + manhattan_distance(cube, i, 0, False) + manhattan_distance(cube, i, 2, False)
    return max(corners / 12, edges / 8)

def convert_exgitform(a):
    x = np.array([])

    for i in ['5','4','1','2','3','6']:
        x = np.append(x, a[i])

    x = x.reshape(18,3)
    x = np.where(x == 0, 'R', x)
    x = np.where(x == '1.0', 'B', x)
    x = np.where(x == '2.0', 'O', x)
    x = np.where(x == '3.0', 'G', x)
    x = np.where(x == '4.0', 'W', x)
    x = np.where(x == '5.0', 'Y', x)
    
    return x

def clockwise(x,n):
    def clockwise1(x): 
        y = np.zeros(9).reshape(3,3)
        y[:,0] = x[2]
        y[:,1] = x[1]
        y[:,2] = x[0]
        return y
        
    for i in range(n):
        x = clockwise1(x)
    
    return x

def counter_clockwise(x,n):
    def counter_clockwise1(x): 
        y = np.zeros(9).reshape(3,3)
        y[:,0] = np.flip(x[0])
        y[:,1] = np.flip(x[1])
        y[:,2] = np.flip(x[2])
        return y
        
    for i in range(n):
        x = counter_clockwise1(x)
    
    return x

def slide_right4ele(a,b,c,d,n):
    x = np.append([np.append(a,b)],[np.append(c,d)])
    x = np.append(x[-n:],x[:-n])
    return x[0:3],x[3:6],x[6:9],x[9:12]
    
def rotate(surface,axis):
    surface = deepcopy(surface)
    if axis == '+x0':
        a,b,c,d = slide_right4ele(surface['1'][0],surface['2'][0],surface['3'][0],surface['4'][0],3)
        surface['1'][0] = a
        surface['2'][0] = b
        surface['3'][0] = c
        surface['4'][0] = d

        surface['5'] = clockwise(surface['5'],3)
    elif axis == '-x0':
        for i in range(3):
            a,b,c,d = slide_right4ele(surface['1'][0],surface['2'][0],surface['3'][0],surface['4'][0],3)
            surface['1'][0] = a
            surface['2'][0] = b
            surface['3'][0] = c
            surface['4'][0] = d

            surface['5'] = clockwise(surface['5'],3)
            
    elif axis == '+x1':
        a,b,c,d = slide_right4ele(surface['1'][1],surface['2'][1],surface['3'][1],surface['4'][1],3)
        surface['1'][1] = a
        surface['2'][1] = b
        surface['3'][1] = c
        surface['4'][1] = d
    elif axis == '-x1':
        for i in range(3):
            a,b,c,d = slide_right4ele(surface['1'][1],surface['2'][1],surface['3'][1],surface['4'][1],3)
            surface['1'][1] = a
            surface['2'][1] = b
            surface['3'][1] = c
            surface['4'][1] = d
            
    elif axis == '+x2':
        a,b,c,d = slide_right4ele(surface['1'][2],surface['2'][2],surface['3'][2],surface['4'][2],3)
        surface['1'][2] = a
        surface['2'][2] = b
        surface['3'][2] = c
        surface['4'][2] = d
        
        surface['6'] = counter_clockwise(surface['6'],3)
    elif axis == '-x2':
        for i in range(3):
            a,b,c,d = slide_right4ele(surface['1'][2],surface['2'][2],surface['3'][2],surface['4'][2],3)
            surface['1'][2] = a
            surface['2'][2] = b
            surface['3'][2] = c
            surface['4'][2] = d

            surface['6'] = counter_clockwise(surface['6'],3)
            
    elif axis == '+y0':
        a,b,c,d = slide_right4ele(np.flip(surface['3'][:,-1]),surface['5'][:,0],surface['1'][:,0],surface['6'][:,0],3)
        surface['3'][:,-1] = np.flip(a)
        surface['5'][:,0] = b
        surface['1'][:,0] = c
        surface['6'][:,0] = d
        
        surface['4'] = counter_clockwise(surface['4'],3)
    elif axis == '-y0':
        for i in range(3):
            a,b,c,d = slide_right4ele(np.flip(surface['3'][:,-1]),surface['5'][:,0],surface['1'][:,0],surface['6'][:,0],3)
            surface['3'][:,-1] = np.flip(a)
            surface['5'][:,0] = b
            surface['1'][:,0] = c
            surface['6'][:,0] = d

            surface['4'] = counter_clockwise(surface['4'],3)
    elif axis == '+y1':
        a,b,c,d = slide_right4ele(np.flip(surface['3'][:,1]),surface['5'][:,1],surface['1'][:,1],surface['6'][:,1],3)
        surface['3'][:,1] = np.flip(a)
        surface['5'][:,1] = b
        surface['1'][:,1] = c
        surface['6'][:,1] = d
    elif axis == '-y1':
        for i in range(3):
            a,b,c,d = slide_right4ele(np.flip(surface['3'][:,1]),surface['5'][:,1],surface['1'][:,1],surface['6'][:,1],3)
            surface['3'][:,1] = np.flip(a)
            surface['5'][:,1] = b
            surface['1'][:,1] = c
            surface['6'][:,1] = d
    
    elif axis == '+y2':
        a,b,c,d = slide_right4ele(np.flip(surface['3'][:,0]),surface['5'][:,2],surface['1'][:,2],surface['6'][:,2],3)
        surface['3'][:,0] = np.flip(a)
        surface['5'][:,2] = b
        surface['1'][:,2] = c
        surface['6'][:,2] = d
        
        surface['2'] = clockwise(surface['2'],3)
    elif axis == '-y2':
        for i in range(3):
            a,b,c,d = slide_right4ele(np.flip(surface['3'][:,0]),surface['5'][:,2],surface['1'][:,2],surface['6'][:,2],3)
            surface['3'][:,0] = np.flip(a)
            surface['5'][:,2] = b
            surface['1'][:,2] = c
            surface['6'][:,2] = d

            surface['2'] = clockwise(surface['2'],3)
    elif axis == '+z0':
        a,b,c,d = slide_right4ele(np.flip(surface['5'][2]),surface['2'][:,0],np.flip(surface['6'][0]),np.flip(surface['4'][:,2]),3)
        surface['5'][2] = a
        surface['2'][:,0] = np.flip(b)
        surface['6'][0] = np.flip(c)
        surface['4'][:,2] = np.flip(d)
        
        surface['1'] = counter_clockwise(surface['1'],3)
    elif axis == '-z0':
        for i in range(3):
            a,b,c,d = slide_right4ele(np.flip(surface['5'][2]),surface['2'][:,0],np.flip(surface['6'][0]),np.flip(surface['4'][:,2]),3)
            surface['5'][2] = a
            surface['2'][:,0] = np.flip(b)
            surface['6'][0] = np.flip(c)
            surface['4'][:,2] = np.flip(d)

            surface['1'] = counter_clockwise(surface['1'],3)
    elif axis == '+z1':
        a,b,c,d = slide_right4ele(np.flip(surface['5'][1]),surface['2'][:,1],np.flip(surface['6'][1]),np.flip(surface['4'][:,1]),3)
        surface['5'][1] = a
        surface['2'][:,1] = np.flip(b)
        surface['6'][1] = np.flip(c)
        surface['4'][:,1] = np.flip(d)
    elif axis == '-z1':
        for i in range(3):
            a,b,c,d = slide_right4ele(np.flip(surface['5'][1]),surface['2'][:,1],np.flip(surface['6'][1]),np.flip(surface['4'][:,1]),3)
            surface['5'][1] = a
            surface['2'][:,1] = np.flip(b)
            surface['6'][1] = np.flip(c)
            surface['4'][:,1] = np.flip(d)
    elif axis == '+z2':
        a,b,c,d = slide_right4ele(np.flip(surface['5'][0]),surface['2'][:,2],np.flip(surface['6'][2]),np.flip(surface['4'][:,0]),3)
        surface['5'][0] = a
        surface['2'][:,2] = np.flip(b)
        surface['6'][2] = np.flip(c)
        surface['4'][:,0] = np.flip(d)
        
        surface['3'] = clockwise(surface['3'],3)
    elif axis == '-z2':
        for i in range(3):
            a,b,c,d = slide_right4ele(np.flip(surface['5'][0]),surface['2'][:,2],np.flip(surface['6'][2]),np.flip(surface['4'][:,0]),3)
            surface['5'][0] = a
            surface['2'][:,2] = np.flip(b)
            surface['6'][2] = np.flip(c)
            surface['4'][:,0] = np.flip(d)
            
            surface['3'] = clockwise(surface['3'],3)
        
    return surface

    
def BX0():
    global start_surface
    print('rotate +X0.. ')
    start_surface = rotate(start_surface,'+x0')
    updateUi_surface(start_surface)
def BLX0():
    global start_surface
    print('rotate -X0.. ')
    start_surface = rotate(start_surface,'-x0')
    updateUi_surface(start_surface) 
def BX1():
    global start_surface
    print('rotate +X1..')
    start_surface = rotate(start_surface,'+x1')
    updateUi_surface(start_surface)
def BLX1():
    global start_surface
    print('rotate -X1..')
    start_surface = rotate(start_surface,'-x1')
    updateUi_surface(start_surface) 
def BX2():
    global start_surface
    print('rotate +X2..')
    start_surface = rotate(start_surface,'+x2')
    updateUi_surface(start_surface)
def BLX2():
    global start_surface
    print('rotate -X2..')
    start_surface = rotate(start_surface,'-x2')
    updateUi_surface(start_surface)
def BY0():
    global start_surface
    print('rotate +Y0..')
    start_surface = rotate(start_surface,'+y0')
    updateUi_surface(start_surface)
def BLY0():
    global start_surface
    print('rotate -Y0..')
    start_surface = rotate(start_surface,'-y0')
    updateUi_surface(start_surface)
def BY1():
    global start_surface
    print('rotate +Y1..')
    start_surface = rotate(start_surface,'+y1')
    updateUi_surface(start_surface)
def BLY1():
    global start_surface
    print('rotate -Y1..')
    start_surface = rotate(start_surface,'-y1')
    updateUi_surface(start_surface)
def BY2():
    global start_surface
    print('rotate +Y2..')
    start_surface = rotate(start_surface,'+y2')
    updateUi_surface(start_surface)
def BLY2():
    global start_surface
    print('rotate -Y2..')
    start_surface = rotate(start_surface,'-y2')
    updateUi_surface(start_surface)
def BZ0():
    global start_surface
    print('rotate +Z0..')
    start_surface = rotate(start_surface,'+z0')
    updateUi_surface(start_surface)
def BLZ0():
    global start_surface
    print('rotate -Z0..')
    start_surface = rotate(start_surface,'-z0')
    updateUi_surface(start_surface)
def BZ1():
    global start_surface
    print('rotate +Z1..')
    start_surface = rotate(start_surface,'+z1')
    updateUi_surface(start_surface)
def BLZ1():
    global start_surface
    print('rotate -Z1..')
    start_surface = rotate(start_surface,'-z1')
    updateUi_surface(start_surface)
def BZ2():
    global start_surface
    print('rotate +Z2..')
    start_surface = rotate(start_surface,'+z2')
    updateUi_surface(start_surface)
def BLZ2():
    global start_surface
    print('rotate -Z2..')
    start_surface = rotate(start_surface,'-z2')
    updateUi_surface(start_surface)
def SOLVE():
    global sol_number
    sol_number = 1
    SOLVEX()
def SOLVE2():
    global sol_number
    sol_number = 2
    SOLVEX()
def SOLVE3():
    global sol_number
    sol_number = 3
    SOLVEX()
    
def SOLVEX():
    global start_surface
    
    #init
    current_surface = {'path': 'str','surface':start_surface}
    msg = 'xxx'
    ever_expand = ''

    u = unit_expand('str')
    limit = 0
    next_limit = 0

    li = []
    if limit == 0:
        for i in u:
            li.append(u[i]['val'])
    li.sort()
    limit  = li[0]
    for i in li:
        if i > limit:
            next_limit = i
            break

    #------------------------------------------
    found_goal = False
    for i in range(1000):
        if found_goal == True:
            break

        print('-'*20)
        for i in u:
            curr_sta = i[:-3]
            break
        msg = 'curr_sta:' + curr_sta + '\tlimit:' + str(limit) + '\tnext limit:' + str(next_limit)
        msg += '\n-------------------------------------------------------------'
        print('curr_sta:',curr_sta,'\tlimit:',limit,'\tnext limit:',next_limit,'\tnext limit:',next_limit)


        if ever_expand == '':
            more_ever = True
        else:
            more_ever = False

        m = {'path':'','val':''}
        for i in u:
            msg += '\n'+ i + '\tval:'+str(u[i]['val']) + '\tg:' + str(u[i]['g']) + '   h:' + str(u[i]['h'])
            print(i,u[i]['val'],'\tg:',u[i]['g'],'h:',u[i]['h'])
            if u[i]['val'] > limit and u[i]['val'] < next_limit:
                next_limit = u[i]['val']

            if u[i]['h'] == 0:
                msg += '\n\nFound Solution -->' + i
                print('found solution--->',i)
                found_goal = True
                break
            if u[i]['val'] <= limit and more_ever == True:
                if m['path'] == '':
                    m['path'] = i
                    m['val'] = u[i]['val']
                elif u[i]['val'] < m['val'] and more_ever == True: 
                    m['path'] = i
                    m['val'] = u[i]['val']
            if i[-3:] == ever_expand:
                more_ever = True
                ever_expand = ''

        if m['path'] != '':
            next_node = m['path']

        else:
            n = [next_node[i:i+3] for i in range(0, len(next_node), 3)]
            ever_expand = n[-1]
            next_node = next_node[:-3]
            next_node

        print('next_node:',next_node)
        if next_node == '':
            print('next limit')
            limit = next_limit
            next_limit = 9999
            next_node = 'str'
            ever_expand = ''

        u = unit_expand(next_node)
        
    
def RANDOM():
    global start_surface
    print('Random surface..')
    
    start_surface = {
        '1' : np.array([0,0,0,0,0,0,0,0,0]).reshape(3,3),
        '2' : np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3),
        '3' : np.array([2,2,2,2,2,2,2,2,2]).reshape(3,3),
        '4' : np.array([3,3,3,3,3,3,3,3,3]).reshape(3,3),
        '5' : np.array([4,4,4,4,4,4,4,4,4]).reshape(3,3),
        '6' : np.array([5,5,5,5,5,5,5,5,5]).reshape(3,3),
    }
        
    #fake random
    for i in ['-x1','+y0']:
        start_surface = rotate(start_surface,i)

#     #real random
#     l = ['+x0','+x1','+x2','+y0','+y1','+y2','-x0','-x1','-x2','-y0','-y1','-y2','-z0','-z1','-z2']
#     for i in range(5):
#         start_surface = rotate(start_surface,random.choice(l))


#init
goal_surface = {
    '1' : np.array([0,0,0,0,0,0,0,0,0]).reshape(3,3),
    '2' : np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3),
    '3' : np.array([2,2,2,2,2,2,2,2,2]).reshape(3,3),
    '4' : np.array([3,3,3,3,3,3,3,3,3]).reshape(3,3),
    '5' : np.array([4,4,4,4,4,4,4,4,4]).reshape(3,3),
    '6' : np.array([5,5,5,5,5,5,5,5,5]).reshape(3,3),
    
}

start_surface = {
    '1' : np.array([0,0,0,0,0,0,0,0,0]).reshape(3,3),
    '2' : np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3),
    '3' : np.array([2,2,2,2,2,2,2,2,2]).reshape(3,3),
    '4' : np.array([3,3,3,3,3,3,3,3,3]).reshape(3,3),
    '5' : np.array([4,4,4,4,4,4,4,4,4]).reshape(3,3),
    '6' : np.array([5,5,5,5,5,5,5,5,5]).reshape(3,3),
    
}

def find_cost(g,s):
    global sol_number
    if sol_number == 1:
        y = find_cost_diff(g,s)
    elif sol_number == 2:
        y = find_cost_ML(g,s)
    else:
        y = find_cost_manh(g,s)
    return y

#Manhaston distance
def find_cost_manh(goal_surface,s):
    c = convert_exgitform(s)
    return corner_edge_sum_max(c)

#ML
def find_cost_ML(goal_surface,surface):
    
    global model
    surface = deepcopy(surface)
    surface2 = deepcopy(surface)
    
    def surfaceToOneArray(surface):
        xx = np.array([])
        for i in range(1,7):
            xx = np.append(xx,surface[str(i)])
        return xx
    def count_zeros(xx):
        count = 0
        for i in xx:
            if i == 0:
                count += 1
        return count

    goal_array = surfaceToOneArray(goal_surface)
    start_array = surfaceToOneArray(surface)
        
    xx = np.subtract(goal_array, start_array) 

    if 54-count_zeros(xx) == 0:
        return 54-count_zeros(xx)
    

    x_surface = surface2

    x = []
    for i in x_surface:
        x += list(x_surface[i].reshape(1,9)[0])

    x = np.array(x).reshape(1, -1)
    y_pred = model.predict(x)    
       
    return int(y_pred[0])

#differance position
def find_cost_diff(goal_surface,surface):
    
    global model
    surface = deepcopy(surface)
    
    def surfaceToOneArray(surface):
        xx = np.array([])
        for i in range(1,7):
            xx = np.append(xx,surface[str(i)])
        return xx
    def count_zeros(xx):
        count = 0
        for i in xx:
            if i == 0:
                count += 1
        return count
    
    goal_array = surfaceToOneArray(goal_surface)
    start_array = surfaceToOneArray(surface)
        
    xx = np.subtract(goal_array, start_array) 

    return (54-count_zeros(xx))/6
    
def update_current_surface(path):
    p = [path[i:i+3] for i in range(0, len(path), 3)]
    sf = start_surface
    for  i in p:
        sf = rotate(sf,i)
        
    s = { 'path':path, 'surface':sf}
    return s
    
def unit_expand(node):

    if not 'str' in node and node != '' and node != None:
        node = 'str'+node
    
    n = [node[i:i+3] for i in range(0, len(node), 3)]
    e = {}
    surface = deepcopy(start_surface)
    for i in ['+x0','+x1','+x2','+y0','+y1','+y2','+z0','+z1','+z2','-x0','-x1','-x2','-y0','-y1','-y2','-z0','-z1','-z2']:
        for ii in n:
            if ii == 'str':
                surface = deepcopy(start_surface)
            else:
                surface = rotate(surface,ii)
        lenge = int(len(node)/3)
        cost = find_cost(goal_surface,rotate(surface,i))
        e.update({node+i:{'val':cost + lenge,'g':lenge,'h':cost}})
        
    return e

def find_min_that_more_threshold(node,threshold,everexplane,next_threshold):
    nod = deepcopy(node)
    m = ''
    mv = 99999
    v = ''
    l = ''
    
    Z = nod.keys()
    if everexplane != '':
        u = unit_expand(everexplane[:-3])
        Z = []
        for i in u.keys():
            if everexplane != 'str' and i != everexplane:
                if u[i]['cost'] > u[everexplane]['cost']:
                    Z.append(i)
                if u[i]['cost'] == u[everexplane]['cost'] and min([i[-3:],everexplane[-3:]]) == everexplane[-3:]:
                    Z.append(i)
            
    for i in Z:
        if nod[i]['cost'] < threshold and nod[i]['cost'] < mv :
            m = i
            mv = nod[i]['cost']
            v = nod[i]['value']
            l = nod[i]['lenge']
        if m == '':
            for i in nod.keys():
                if nod[i]['cost'] >= threshold:
                    if next_threshold == threshold:
                        next_threshold = nod[i]['cost']
                    if nod[i]['cost'] < next_threshold and nod[i]['cost'] > threshold:
                        next_threshold = nod[i]['cost']
  
    return m,mv,v,l,next_threshold

def rand(rng):
    global start_surface
    
    rng_serface = start_surface
    l = ['+x0','+x1','+x2','+y0','+y1','+y2','-x0','-x1','-x2','-y0','-y1','-y2','-z0','-z1','-z2']
    for i in range(rng):
        ll = random.choice(l)
        rng_serface = rotate(rng_serface,ll)
    return rng_serface

def prepare_data(n): 
    with open('rubik18_data.csv', 'w') as f:
        write = csv.writer(f)
        for i in range(n):
            y = random.randint(1,70)
            x_surface = rand(y)
            x = []
            for ii in x_surface:
                x += list(x_surface[ii].reshape(1,9)[0])
            x.append(y)
            write.writerow(x)
            print(i)

sol_number = int(input('Enter Solver method 1.Diiferent positioin  2.Machine Learning  3.Manhuston Distance >>>'))
move = input('Enter move Rubik ex. -x1+y0 >>> ')
move = [move[i:i+3] for i in range(0, len(move), 3)]

for i in move:
    start_surface = rotate(start_surface,i)
        
SOLVEX()

    
    
    
    
    
