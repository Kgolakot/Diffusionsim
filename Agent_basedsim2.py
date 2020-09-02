# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:38:44 2020

@author: kgolakot
"""

import random
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import truncnorm

#parameter initialisation for the social netwrok   
gamma = 1.75
nodes = 5000
xmin = 1

#generate degree through power law dsitribution
def powerlaw(xmin, u_rv, gamma, xmax):
    #p_rv = xmin*(1 -u_rv)**(1/1-gamma)
    a = xmax**(1-gamma)
    b = xmin**(1-gamma)
    p_rv = ((a - b)*u_rv + b)**(1/1-gamma)
    return p_rv

#assign the nodes based on the degree
def gen_node(m):
    p_rv = []
    for i in range(nodes):
        u_rv = random.uniform(0,1)
        temp = powerlaw(xmin, u_rv, gamma, nodes)
        temp = round(temp)
        p_rv.append(temp)
    return p_rv

#fun to generate random nodes
def ugen(m, k):
    s = random.sample(range(m), k)
    return s

#generate the degree list
degree_list = gen_node(nodes)
m = sum(degree_list)

#initialize the matrix
node_list = np.zeros((1, 2))
a = np.zeros((1,2))

#generate the top right of the matrix
for i in range(nodes):
  b = np.zeros((degree_list[i], 2))
  temp = ugen(nodes, degree_list[i])
  for j in range(degree_list[i]):
    b[j,0] = i
    b[j,1] = temp[j] 
  node_list = np.concatenate((node_list,b))

node_list = np.delete(node_list,0,0)
node_list = node_list.astype(int)

#generate the adjacency matrix        
soc_net = np.zeros((nodes, nodes))

for i in range(len(node_list)):
    soc_net[node_list[i,0],node_list[i,1]] = 1
    
soc_net = soc_net + np.transpose(soc_net)

for i in range(nodes):
    for j in range(nodes):
        if(soc_net[i,j]>1):
            soc_net[i,j] = 1

#visulaize the adjacency matrix
fig = plt.figure(figsize = (5,5))
plt.imshow(soc_net, cmap="Greys", interpolation="none")

cons_type = {0: 'Innovator', 1: 'Early adopter', 2: 'Early Majority', 3:'Late majority', 4:'Laggards'}

cons_values = [0, 1, 2, 3, 4]
#cdf distribution of agents depending on influence
p_cons = [0.975, 0.84, 0.50, 0.16]
node_lim = []

#find the node lim depending on agent distribtuion of their influence    
def node_dist(p_cons):
    node_lim = []
    for i in range(len(p_cons)):
        temp = powerlaw(xmin, p_cons[i], gamma, nodes)
        temp = round(temp)
        node_lim.append(temp)
    return node_lim


node_lim = node_dist(p_cons) 
agent_type = np.zeros((nodes))
agent_type.fill(5)

#assign agents their level of influence
def assign_type(agent, lim, nodes, deg):
    for i in range(nodes):
      j = 0
      while j<3:
         if(deg[i] >= lim[j]):
              agent[i] = j
              break
         else: 
             j=j+1
      if(agent[i] == 5):
          b_rv = np.random.binomial(1, 0.32, 1) 
          u_rv = random.uniform(0,1)
          if(b_rv<u_rv):
              agent[i] = 3
          else: 
              agent[i] = 4      
    return agent
    
agent_type = assign_type(agent_type, node_lim, nodes, degree_list)    
temp_type = agent_type 

#count the number fo different agent tyoes
count = [0, 0, 0, 0, 0]
for i in range(nodes):
    for j in range(5):
        if (agent_type[i] == j):
            count[j] = count[j] + 1
            

weight_list = np.zeros((1, 3))
adap_list = np.zeros((1,3))

#create the matrix of different agent types
for i in range(nodes):
    w_rv = np.random.uniform(0,1,3)
    a_rv = np.random.uniform(0,1,3)
    temp_rv = np.zeros((1,3))
    temp2_rv = np.zeros((1,3))
    temp_rv[0,0] = w_rv[0]/sum(w_rv)
    temp_rv[0,1] = w_rv[1]/sum(w_rv) 
    temp_rv[0,2] = w_rv[2]/sum(w_rv)
    temp2_rv[0,0] = a_rv[0]/sum(a_rv)
    temp2_rv[0,1] = a_rv[0]/sum(a_rv)
    temp2_rv[0,2] = a_rv[0]/sum(a_rv)
    weight_list = np.concatenate((weight_list,temp_rv))
    adap_list = np.concatenate((adap_list, temp2_rv))
         
weight_list = np.delete(weight_list, 0, 0)
adap_list = np.delete(adap_list, 0, 0)   

#assign brand factors
brand_name = {1:"Google", 2:"Samsung", 3:"Xiaomi", 4:"Others"}
battery = [0.9, 1, 0.95, 0.78]
physical_appearance = [0.9, 1, 0.8, 0.65]
innovative = [0.99, 0.95, 0.82, 0.65]
price = [0.95, 0.999, 0.86, 0.696]

#perceieved product quality
ppq = np.zeros((1, 4))

#calculate perceived product quality
for i in range(nodes):
    allot = np.zeros((1,4))
    for j in range(4):
        allot[0,j] = weight_list[i,0]*battery[j] + weight_list[i,1]*physical_appearance[j] + weight_list[i,2]*innovative[j]
    ppq = np.concatenate((ppq, allot))    

# calculate node index in adjacecny array
def node_index(d_list, pos):
    index = 0
    for i in range(pos):
        index = index + d_list[i]
    return index

#calculate the influence of other neighbouring agents on the current agent who is making a decision to buy
def calculate_influence(n_list, agent_num, degree, b_type, ast_type):
    n_index = node_index(degree, agent_num)
    neighbour = []
    for i in range(degree[agent_num]):
        t = n_list[n_index + i, 1]
        neighbour.append(t)
    influence = []
    for i in range(len(brand_name)):
        div = 0
        n = 0
        for j in range(len(neighbour)):
            omega = 0
            if(b_type[neighbour[j]] == i):
                z = abs(ast_type[agent_num] - ast_type[neighbour[j]])
                if(z == 0):
                    omega = 1
                elif(z == 1):
                    omega = 0.8
                elif(z == 2):
                    omega = 0.6
                elif(z == 3):
                    omega = 0.4
                else:
                    omega = 0.2
                n = n + omega
                div = div + 1
        if(div == 0):
            influence.append(0)
        else:
            k = n/div
            influence.append(k)    
    return influence                   

#find the agents who are going to become the next potential buyers       
def who_next(arrival, ast_type):
    index = []
    adopt = []
    t_type = np.copy(ast_type)
    for i in range(4):
        if(arrival == 0):
            break
        else:
            q = np.where(t_type == i)
            index = list(q[0])
            if(len(index) == 0):
                continue
            temp_index = random.choice(index)
            adopt.append(temp_index)
            t_type[temp_index] = 10
            index.remove(temp_index)
            arrival = arrival - 1
    return adopt   

#find the number of arrivals
def arrival():
    time_limit = 0
    n_arrival = 0
    mu = 2
    for i in range(0, 100):
        temp = -math.log(1 - random.uniform(0,1))/mu
        time_limit =  time_limit + temp
        n_arrival = n_arrival + 1
        if(time_limit > 7):
            break
    return n_arrival

#assign utility threshold
def utility_threshold():
    #u_t = random.uniform(0.1, 0.8)
    u_t = truncnorm.rvs(-1, 0, 1)
    return u_t 

time_step = 0
ppq = np.delete(ppq, 0, 0)
adopted_brand = np.zeros((nodes, 1))
adopted_brand.fill(7)
adopted_status = np.zeros((nodes, 1))

agent_track = np.arange(0, nodes)
agent_track = agent_track.tolist()
time_step = 0
current_share = np.zeros((1,4))
brand_sales = np.zeros((1,4))
present_sales = np.zeros((1,4))

#Agent simulation
for i in range(1, 20000):
    if(len(agent_track) == 0):
        break
    adopter = []
    ar = arrival()
    adopter = who_next(ar, temp_type)
    temp_share = np.zeros((1,4))
    temp_sales = np.zeros((1,4))
    record_sales = np.zeros((1,4))
    for i in range(len(adopter)):
        if(adopted_status[adopter[i], 0] == 1):
            continue
        brand_util = []
        inf = []
        inf = calculate_influence(node_list, adopter[i], degree_list, adopted_brand, temp_type)
        for j in range(len(brand_name)):
            k = adap_list[adopter[i],0]*(1 - price[j]) + adap_list[adopter[i],1]*ppq[adopter[i], j] + adap_list[adopter[i],1]*inf[j]
            brand_util.append(k)
        max_brand = 0
        max_brand = max(brand_util)
        util = 0
        util = utility_threshold()
        if(max_brand > util):
            adopted_brand[adopter[i]] = brand_util.index(max_brand)
            temp_share[0, brand_util.index(max_brand)] += 1
            adopted_status[adopter[i], 0] = 1
            temp_type[adopter[i]] = 10
            record_sales[0, brand_util.index(max_brand)] += 1 
            print(adopter[i])
            agent_track.remove(adopter[i])
    k = np.sum(temp_share)
    if(k == 0):
        k = 1
    for i in range(len(brand_name)):
        temp_share[0, i] = temp_share[0, i]/k
        y = np.where(adopted_brand == i)
        temp_sales[0, i] = len(y[0])
    present_sales = np.concatenate((present_sales, record_sales))
    current_share = np.concatenate((current_share, temp_share))
    brand_sales = np.concatenate((brand_sales, temp_sales))

#iteration = np.arange(0, 20000)           
#plt.plot(iteration, brand_sales[:,0], 'r--', iteration, brand_sales[:,1], 'bs', iteration, brand_sales[:,2], 'g^', iteration, brand_sales[:,3], 'ro')
#plt.show()
    
#plt.plot(iteration, present_sales[:,0], 'r--', iteration, present_sales[:,1], 'bs', iteration, present_sales[:,2], 'g^', iteration, present_sales[:,3], 'ro')
#plt.show()
              
#plt.stackplot(iteration , current_share.T, labels=['A','B','C', 'D'])
#plt.legend(loc='upper left')
#plt.show()     
    
brands = ['Google', 'Samsung', 'Xiaomi', 'Others'] 
sales = brand_sales[19999, :].tolist()
y_pos = np.arange(len(brands))
plt.bar(y_pos, sales, color = (0.5, 0.1, 0.5, 0.5))
plt.xticks(y_pos, brands)
plt.ylim(0, 1800)
plt.show()


sizes = [sales[0]/sum(sales), sales[1]/sum(sales), sales[2]/sum(sales), sales[3]/sum(sales)]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=brands, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
 
    