import numpy as np
from scipy.stats import chisquare

length_grid=577
rows_grid=24  #sqrt(length_grid-1)
cols_grid=24
grid=np.arange(1, length_grid, 1, dtype=int)
grid=grid.reshape((cols_grid, rows_grid))

#predict using prediction module and use path to data here
res_G = np.load('res_G.npy', allow_pickle=True).tolist()
target_G = np.load('target_G.npy', allow_pickle=True).tolist()
res_D = np.load('res_D.npy', allow_pickle=True).tolist()
target_D = np.load('target_D.npy', allow_pickle=True).tolist()



def first_time():
 time_l=[]
 folder=[]
 for x in range(0,1247):
  time_instant=np.load('add path here' +str(x) +'/time.npy', allow_pickle=True).tolist()

  time_l.append(time_instant[0])
  folder.append(x)
 return time_l,folder


time_l,folder=first_time()


res_G = np.asarray(res_G, dtype=np.float32)

res_G = res_G.reshape(576, 576)


target_G = np.asarray(target_G, dtype=np.float32)
target_G = target_G.reshape(576, 576)

map_f=zip(time_l,folder)
map_f=dict(map_f)

map_r=zip(folder,time_l)
map_r=dict(map_r)


#Mapping
relative=[]
for i in range(0,cols_grid):
  for j in range(0,rows_grid):
    relative.append(tuple([i,j]))
positional=list(range(0, rows_grid*cols_grid+1,1))
map=zip(relative,positional)
map=dict(map)  #maps coordinates to grid cell

map1=zip(positional,relative)  # maps grid cells to coordinate
map1=dict(map1)


sav=res_G



#calculates shortest path
def SP(source,des):

 a,b=source
 c,d=des
 shortest_path=[source]
 row=abs(c-a) #move these many rows down
 col=abs(d-b)
 if c!=a:
  row_dir=int(row/(c-a))
 else:
  row_dir=1
 if d!=b:
  col_dir=int(col/(d-b))
 #find min(row,col)
 else:
   col_dir=1
 if row<=col:
   eql_move=row
   #row=0   #we dont need to move any more rows now
 else:
   eql_move=col
  # col=0
 row=row-eql_move
 col=col-eql_move
 for i in range(1,eql_move+1): #adds (2,2),(3,3),...
  a=a+row_dir
  b=b+col_dir
  shortest_path.append(tuple([a,b]))

 for j in range(0,col):
   b=b+col_dir
   shortest_path.append(tuple([a,b]))
 for j in range(0,row):
   a=a+row_dir
   shortest_path.append(tuple([a,b]))


 return shortest_path


def len_sp(source,dest):
 a,b=source
 c,d=dest
 shortest_path=[source]
 row=abs(c-a) #move these many rows down
 col=abs(d-b)
 if c!=a:
  row_dir=int(row/(c-a))
 else:
  row_dir=1
 if d!=b:
  col_dir=int(col/(d-b))
 #find min(row,col)
 else:
   col_dir=1
 if row<=col:
   eql_move=row
   #row=0   #we dont need to move any more rows now
 else:
   eql_move=col
  # col=0
 row=row-eql_move
 col=col-eql_move
 for i in range(1,eql_move+1): #adds (2,2),(3,3),...
  a=a+row_dir
  b=b+col_dir
  shortest_path.append(tuple([a,b]))

 for j in range(0,col):
   b=b+col_dir
   shortest_path.append(tuple([a,b]))
 for j in range(0,row):
   a=a+row_dir
   shortest_path.append(tuple([a,b]))

 length=len(shortest_path)-1
 return length


import sys
def findMax(arr):
    max = arr[0]
    n=len(arr)  #number of rows

    # Traverse array elements from second
    # and compare every element with
    # current max
    for i in range(1, n):
        if arr[i] > max:
            max = arr[i]
    return max


import math
def dist(cpx,cpy,gr):
 dist=[]
 cor=[]
 for i,j in zip(gr[0],gr[1]):
  if i!=j:  #dont loop over same grid cell
    x,y=map1[i]
    dist.append(tuple([math.sqrt((cpx-x)**2+(cpy-y)**2)]))
    cor.append(tuple([x,y]))


 d_m=min(dist)
 index=dist.index(d_m)
 val= cor[index]


 return val

n=1000
loc_x=np.random.randint(0, high=23, size=n, dtype=int)
loc_y=np.random.randint(0, high=23, size=n, dtype=int)
loc= list(zip(loc_x,loc_y))


capacity=2
gr=np.where(res_G>0)

def DAG(res_G,source,capacity):  #will take as input full request(predicted) graph and generate DAG for current source
 #source is of form row index col index like (2,7)
 path=[]
 arr=[]
 dest_arr=[]
 br=0  #variable to break out of loop
 dag_arr=-100*np.ones((576, 576))

 sx,sy=source # x and y coordinates of source
 src=map[source]
 tru=0
 if sum(np.array(res_G[src]))==0: # or tru==1: #if there are no requests from current point to dest relocate drievr through shortest path to demand aware location
  val=dist(sx,sy,gr)
  #val conatins src and dest
  #val_s,val_d=val
 #move from shortest path from src to val and update src
  path=SP(source,val)
  src=map[val]
  sx,sy=val # x and y coordinates of source
 dest=0
 res_G[src][src]=0  #cant loop over src
 dest=findMax(np.array(res_G[src]))

 dest=np.where(res_G[src]==dest)[0][0]
 tempor2=min(capacity,res_G[src][dest])
 res_G[src][dest]=res_G[src][dest]- tempor2

 capacity=capacity-tempor2
 destination=map1[dest]
 dx,dy=destination
 #create DAG between source and destination
 r=sx-dx
 c=sy-dy

 for i in range(0,abs(r)+1):
  for j in range(0,abs(c)+1):
   if r!=0:
    dx=int(r/abs(r)) #direction to move
   else:
     dx=0
   if c!=0:
    dy=int(c/abs(c)) #direction to move
   else:
     dy=0
   arr.append(tuple([int(sx-dx*i),int(sy-dy*j)]))


 for f in arr:
     mp=map[f]
     x,y=f
     if (x-dx,y) in arr:
       dag_arr[map[(x,y)]][map[(x-dx,y)]]=res_G[map[(x,y)]][map[(x-dx,y)]]
     if (x,y-dy) in arr:
       dag_arr[map[(x,y)]][map[(x,y-dy)]]=res_G[map[(x,y)]][map[(x,y-dy)]]
     if (x-dx,y-dy) in arr:
       dag_arr[map[(x,y)]][map[(x-dx,y-dy)]]=res_G[map[(x,y)]][map[(x-dx,y-dy)]]
 return dag_arr,arr,r,c,destination,capacity,src,path


alpha=1.5
sps=np.zeros(length_grid) #generates requests matrix of length length_grid*length_grid. Each element of matrix is randomly generated between 0 and 6

#find shortest path from all vertices to destination
def sp(dest):
  a,b=dest
  for i in range(0,length_grid-1):  #could use optimized destination in place of length_grid
      sps[i]=len_sp(map1[i],dest)
  return sps

import math

import math
def dp(r,c,src,dag,vertices,req_t,distances,capacity):

  for i in range(0,abs(r)+1):
   for j in range(0,abs(c)+1):
    req=[]
    reqs=[]  #temporary variables
    if i==0 and j==0:
      req_t[0][0]=0
      vertices[0][0]=0
      distances[0][0]=0
    else:
     if i-1>=0:
      x=0
      x=alpha*sps[src]-sps[dag[i][j]] -distances[i-1][j]#[dag[i-1][j]]
      if x>=1:
        req.append(dag[i-1][j])
        reqs.append(res_G[dag[i-1][j]][dag[i][j]] + req_t[i-1][j])  #append req_t here

     if j-1>=0:
      x=alpha*sps[src]-sps[dag[i][j]] -distances[i][j-1]#[dag[i][j-1]]
      if x>=1:
        req.append(dag[i][j-1])
        reqs.append(res_G[dag[i][j-1]][dag[i][j]] + req_t[i][j-1])
     if i-1>=0 and j-1>=0:
      x=alpha*sps[src]-sps[dag[i][j]] -distances[i-1][j-1]#[dag[i-1][j-1]]
      if x>=1:
        req.append(dag[i-1][j-1])
        reqs.append(res_G[dag[i-1][j-1]][dag[i][j]] + req_t[i-1][j-1])
     if req:
      mx=max(reqs)
      ind=reqs.index(mx)
      el=req[ind]
      vertices[i][j]=el
      tempor=min(capacity,res_G[el][dag[i][j]])
      res_G[el][dag[i][j]]= res_G[el][dag[i][j]]-tempor
      capacity=capacity-tempor
      z=list(zip(*np.where(dag==el)))
      a=z[0][0]
      b=z[0][1]
      req_t[i][j]=mx
      distances[i][j]= distances[a][b]+1
     else:  #make this vertex unreachable
       req_t[i][j]=-100
       distances[i][j]= 100
       vertices[i][j]=-100
  return vertices,req_t,distances,capacity

def ret_path(vertices,src):
 dpath=[map[dest]]
 x=list(zip(*np.where(dag==map[dest])))
 a=int(vertices[x[0][0],x[0][1]])
 while a!=src:
  dpath.append(a)
  x=list(zip(*np.where(dag==a)))
  a=int(vertices[x[0][0],x[0][1]])

 return dpath

alpha=1.5

def req_in_path(path):   #check detour ratiio here>>>>
  #use actual req set and see how many request does current path cover
  # we need to know requests origin and destination
  rq_p=[]
  rq_comp=[]

  for i in range(0,len(path)):
    for j in range(i,len(path)):
      if target_G[path[i]][path[j]]:
        rq_p.append(tuple([path[i],path[j]]))
  for i in rq_p:
    a,b=i
    if a==b:
      rq_comp.append(tuple([a,b]))

    if a!=b:
     if ((path.index(b)-path.index(a))/len_sp(map1[a],map1[b]))<=random.uniform(1,2):
      rq_comp.append(tuple([a,b]))
  return rq_comp

def compatible_reqs(path,req_path):
  capacity=2
  edges={}
  taken_reqs=[]

  for first, second in zip(path, path[1:]):
     edges[(first, second)] = 0 #initially all edges have 0 requests
  br=0
  temp=[]
  temp2=[]
  reqs_s=[]
  cv=[]
  for c in req_path:
      temp.append(math.sqrt((c[1]-c[0])**2))
      cv.append(math.sqrt((c[1]-c[0])**2))
  temp2=temp
  temp2.sort(reverse=True)
  arr=np.argsort(np.array(temp))
  brr=list(arr[::-1])
  if brr:
     for k in brr:
      reqs_s.append(req_path[k])
  for i in reqs_s:
   br=0
   a,b=i
   ind1=path.index(a)
   ind2=path.index(b)
   for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
      edges[(first,second)]=edges[(first,second)]+target_G[path[ind1]][path[ind2]]
   for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
    if edges[(first,second)]>capacity:
      br=1
      for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
       edges[(first,second)]=edges[(first,second)]-target_G[path[ind1]][path[ind2]]

   if br==0:  #req can be taken
    taken_reqs.append(i)
    m,n=i
    target_G[m][n]=0  #as all reqs are taken #target_G[m][n]-capacity
  return taken_reqs


def fare_cal(taken_reqs):
  fare=[]
  bf=2.55 #base fare
  p_m=0.35
  p_mile=1.75
  far=0
  fr=0
  mf=7
  #1 litre=1.01 USD can cover 12.5km
  #2.5km can be done in 0.2 litres which implies 0.202 $
  #15 minutes for 2.5km
  #avg speed=3mph   in northnumber avenue #https://www.forbes.com/sites/carl5/22/uber-data-reveals-motoring-slower-than-walking-in-many-cities/?sh=5c35f71c16fb
  #0.129 petrol per mile as 2.5km is 1.55 miles
  #.129*1.01 is price of petrol for .129 litres
  if len(taken_reqs)>1:
   for i in taken_reqs:
    dist=(len_sp(map1[i[0]],map1[i[1]])+1)*1.24 #+1.55  #as we have done +1 so it counts the grid also
    time=dist/0.05 #3 miles per hour is 0.05 miles per minute
    fr=bf+p_m*time +p_mile*dist
    if fr<mf:
     far=mf
    else:
     far=fr
    fare.append(0.8*far)
  else:
   dist=(len_sp(map1[taken_reqs[0][0]],map1[taken_reqs[0][1]])+1)*1.24 #+1.55
   time=dist/0.05
   fr=bf+p_m*time +p_mile*dist
   if fr<mf:
     far=mf
   else:
     far=fr
   fare.append(far)
  return sum(fare)




def dist_of_path(path,distances,dag,dest):
  a=list(zip(*np.where(dag==dest)))
  if path:
    return (distances[a[0][0]][a[0][1]]+len_sp(path[0],path[len(path)-1]))*1.24 +1.24
  else:
   return (distances[a[0][0]][a[0][1]])*1.24+1.24


def normalize(array):

  if np.min(array) < 0:
    array -= np.min(array)
  return array


def categorize_area():
  requests_grid=[]
  target_grid=[]
  area_sparse={}
  k=1
  for i in range(0,576):
    requests_grid.append(sum(res_G[i]))
    target_grid.append(sum(target_G[i]))
  area_s=zip(relative,target_grid)
  area_s=dict(area_s)
  for i in range(0,24):
    for j in range(0,24):
      reqs_total=0
      for a in range(0,k):
        for b in range(0,k):
          if area_s.get((i+a,j+b))!=None:
            reqs_total= reqs_total+ area_s[(i+a,j+b)]

      for a in range(1,k):
        for b in range(0,k):
          if area_s.get((i-a,j+b))!=None:
            reqs_total= reqs_total+ area_s[(i-a,j+b)]
      for a in range(0,k):
        for b in range(1,k):
          if area_s.get((i+a,j-b))!=None:
            reqs_total= reqs_total+ area_s[(i+a,j-b)]
      for a in range(1,k):
        for b in range(1,k):
          if area_s.get((i-a,j-b))!=None:
           reqs_total= reqs_total+ area_s[(i-a,j-b)]
      if reqs_total<=0:
         area_sparse[(i,j)]=True
      else:
        area_sparse[(i,j)]=False
  return area_sparse,area_s
area_sparse,area_s=categorize_area()  #area is considered as sparse if its 3 hop neighbors dont contain any request
#it is used for making non-myopic decision


def calc_rel(location,hops):
    i,j=location
    area_r=[]
    req_area=[]
    for a in range(0,hops):
        for b in range(0,hops):
          if area_s.get((i+a,j+b))!=None:
            if area_s.get((i+a,j+b))>0:
               area_r.append((i+a,j+b))
               req_area.append(area_s.get((i+a,j+b)))
    for a in range(1,hops):
        for b in range(0,hops):
          if area_s.get((i-a,j+b))!=None:
            if area_s.get((i-a,j+b))>0:
               area_r.append((i-a,j+b))

               req_area.append(area_s.get((i-a,j+b)))
    for a in range(0,hops):
        for b in range(1,hops):
          if area_s.get((i+a,j-b))!=None:
              if area_s.get((i+a,j-b))>0:
               area_r.append((i+a,j-b))

               req_area.append(area_s.get((i+a,j-b)))
    for a in range(1,hops):
        for b in range(1,hops):

          if area_s.get((i-a,j-b))!=None:
            if area_s.get((i-a,j-b))>0:
               area_r.append((i-a,j-b))

               req_area.append(area_s.get((i-a,j-b)))

    return area_r,req_area

def relocate(location,hops):


    in_l=0
    area_r=[]
    req_area=[]

    area_r,req_area=calc_rel(location,hops)
    sorted_area=[0]*(10000)
    sorted_req_area=[0]*(10000)
    while len(req_area)==0:
       area_r,req_area=calc_rel(location,hops+1)

    indices=np.argsort(np.array(req_area))
    k=0
    for i in indices[::-1]:
      print(area_r[i])
      sorted_area[k]=area_r[i]
      sorted_req_area[k]=req_area[i]
      k=k+1
    for i in sorted_area:
      if area_sparse[i]==False:
        in_l=in_l+1
        return i
      elif in_l>=len(sorted_area):
        return sorted_area[0]




import random
def WT(path,reqs):
  wt=[]
  for i in reqs:
    s,d=i
    t_s=path.index(s)-random.randint(int(path.index(s)*3/4), path.index(s))

    wt.append(t_s*1.24/0.05)
  return wt

def cal_num(path,start,end):
  start_index = path.index(start)
  end_index=path.index(end)
  return abs(end_index-start_index)


def time_gr(loc1,loc2):
  dist=len_sp(loc1,loc2)
  dist=dist*1.24   #miles
  tim=dist/3 #time in hours
  tim=tim*60
  return math.ceil(tim)





def loc_sparsity_check(res_list,com_reqs,taken_reqs_matrix,target_G1):
  for i,j in com_reqs:
    taken_reqs_matrix[i][j]= taken_reqs_matrix[i][j]+target_G1[i][j]
  return taken_reqs_matrix

def find_elements_less_than_one_third_sorted(arr):
    # Flatten the 2D array into a 1D array
    flat_arr = arr.copy()

    if len(flat_arr) == 0:
        return []  # No positive integers found

    flat_arr.sort()  # Sort the array

    index = len(flat_arr) // 3  # Calculate the 1/3rd position index

    if index >= len(flat_arr):
        return []  # Index out of range

    one_third_element = flat_arr[index]  # Get the 1/3rd smallest positive integer

    # Return all elements less than the 1/3rd smallest positive integer
    less_than_one_third = [num for num in flat_arr if num < one_third_element]

    return flat_arr[:1000]#less_than_one_third


def find_k_elements(la,loc_sparsity_arr,x,y,k):
    xarr=[]
    yarr=[]
    da=la.copy()
    da.sort()
    da=da[:k]
    az=0

    a=la.copy()
    b=da.copy()
    az = 0
    ind = 0
    dup = {}
    indices = []
    for i in b:
        if i in dup:
            ind = dup[i]
            ind = ind + a[ind:].index(i) + 1  # Search for the element 'i' in the remaining portion of 'a'
            indices.append(ind - 1)
            dup[i] = ind
            az = az + ind  # Update the starting index for the next search
        else:
            ind1 = a.index(i)  # Search for the element 'i' in the remaining portion of 'a'
            #   print("i,ind1",i,ind1)

            dup[i] = ind1 + 1
            indices.append(ind1)
    for i in indices:
        xarr.append(x[i])
        yarr.append(y[i])
    return xarr, yarr

def loc_sparsity(reqs_matrix):

 loc_sparsity_arr=reqs_matrix.copy()
 for i in range(0,576):
   for j in range(0,576):
    if reqs_matrix[i][j]!=0:
      loc_sparsity_arr[i][j]=taken_reqs_matrix[i][j]/reqs_matrix[i][j]

 x=[]
 y=[]
 for i in range(0,576):
     for j in range(0,576):
         if reqs_matrix[i][j]>0:
             x.append(i)
             y.append(j)
 la=[]
 lb=[]
 for i,j in zip(x,y):

       la.append(loc_sparsity_arr[i][j])
       lb.append(reqs_matrix[i][j])
 elements = find_elements_less_than_one_third_sorted(np.array(la))
 k1=1400

 xind,yind=find_k_elements(la,loc_sparsity_arr,x,y,k1)
 return  elements,la,x,y,xind,yind,lb

import statistics


from fractions import Fraction

def generate_fraction(*args):
    denominator = all

    numerator = dense
    return Fraction(numerator, denominator)



def nodup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def sort_drivers(d,dense,all):
  drivers=[]
  init_loc=np.random.randint(0, high=n, dtype=int)
  loc_reqs=[]
  for item1,item2 in zip(dense,all):
    if item2!=0:
     loc_reqs.append(item1/item2)
    else:
      loc_reqs.append(0)

  while loc_reqs[init_loc]==0:
   init_loc=np.random.randint(0, high=n, dtype=int)
  prop_loc=init_loc
  count=0
  for i in range(0,35):
   if init_loc==prop_loc:
    drivers.append(init_loc)
    count=count+1
   prop_loc = int(random.uniform(0, n))
   while loc_reqs[prop_loc]==0:
    prop_loc=int(random.uniform(0, n))
   if loc_reqs[init_loc]!=0:
    a_p=loc_reqs[prop_loc]/loc_reqs[init_loc]
   else:
     a_p=0
   if a_p>=1:
    init_loc=prop_loc

   else:


    r = random.random()
    if a_p>=r:
     init_loc=prop_loc

    else:
     init_loc=init_loc


  matrix2=loc_reqs
  drivers_noduplicate = nodup(drivers)#[*set(drivers)]
  if len(drivers_noduplicate)>0:
   return drivers_noduplicate[::-1],loc_reqs
  else:
    return d,loc_reqs







def kl_divergence(p, q):
    """Calculate KL divergence between two discrete distributions."""
    epsilon = 1e-9#1e-9  # A small epsilon value to avoid division by zero
    q = np.where(q == 0, epsilon, q)  # Replace 0s in q with epsilon
    p = np.where(p == 0, epsilon, p)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def driver_kl_diver(driver_loc_arr,sparse):
    a1 = []
    b1 = []
    for i in driver_loc_arr:
        a1.append(i / sum(driver_loc_arr))

    probabilities = np.array(a1)

    # Calculate the expected frequencies
    exp = [n / sum(sparse)] * (n)
    for i in exp:
        b1.append(i / sum(exp))

    expected_prob = np.array(b1)
    kl=kl_divergence(probabilities,expected_prob)
    print("KL Divergence driver:", kl)


def kl_rider(loc_sparsity_arr,reqs,num): #reqs are total reqs in the area it doesnt take taken requests into account
    a1 = []
    b1 = []
    sum_of_elements = np.sum(reqs)
    normalized_array = reqs / sum_of_elements
    normalized_array=normalized_array*num
    normalized_array = np.round(normalized_array)


    while np.sum(normalized_array)>num:
        jk=random.randint(0,len(normalized_array)-1)

        normalized_array[jk]=max(normalized_array[jk]-1,1)

    reqs_final=normalized_array/reqs


    kl = kl_divergence(np.array(loc_sparsity_arr), np.array(reqs_final))
    print("KL Divergence rider:", kl)


import itertools

reqs_matrix=np.zeros((576, 576), dtype=np.float32)
taken_reqs_matrix= np.zeros((576, 576), dtype=np.float32)

n=30
n_i=32
fr=[0]*(n)
fr_s=[0]*(n)
distan=[0]*(n)
futility=[0]*(n)
fwt=[0]*(n*n_i)


dense=[0]*(n)
sparse=[0]*(n)
all=[0]*(n)
s_t=[0]*(n)

d=list(range(0,n))  #driver indices
d_s=list(range(0,n))

o=0
fu_ci=[0]*(n_i)
fu_i=0

ppg=[0]*(n)
ride_o=[0]*(n)
ride_wo=[0]*(n)
loc_x=np.random.randint(0, high=23, size=n, dtype=int)
loc_y=np.random.randint(0, high=23, size=n, dtype=int)
loc= list(zip(loc_x,loc_y))#np.array(zip(loc_x,loc_y))

ol=0
cad=[]
wt_avg=0
eff_avg=0
fol=1158
sparse_total=0
up=[fol]*(n)
prev_up=[fol]*(n)
loc_cou=0
fi=0

for cd in range(1,n_i):

   target_G = np.load('path to file' +str(fol) + '/target_G.npy', allow_pickle=True).tolist()
   target_D = np.load(
       'path to file' + str(
           fol) + '/target_D.npy', allow_pickle=True).tolist()

   target_G = np.array(target_G)
   target_G = target_G.reshape(576, 576)
   reqs_matrix=np.add(reqs_matrix,target_G)

   target_D = np.array(target_D)
   target_D = target_D.ravel()

   count_greater_than_zero = np.sum(target_D > 0)
   count_zero = np.sum(target_D == 0)

   t_s=map_r[fol]
   t_s=t_s+1
   fol=map_f[t_s]
for cd in range(1,n_i): #see n drivers earn over n_i time periods
 o=0

 for ak in prev_up:
  target_G = np.load('path to file' +str(ak) + '/target_G.npy', allow_pickle=True).tolist()
  res_G = np.load('path to file' +str(ak) + '/res_G.npy', allow_pickle=True).tolist()
  target_G = np.asarray(target_G, dtype=np.float32)
  target_G = target_G.reshape(576, 576)


  np.save('use different path for data modification' +str(ak) + '/res_G.npy', res_G)#.detach().numpy())
  np.save('use different path for data modification' +str(ak) + '/target_G.npy', target_G)#.detach().numpy())


 zk=[]
 r_a=1
 driver_i=0
 for aq in loc:
   zk.append(map[aq])

 if fi == 0:
     d_s = d
     fi = 1
 else:
     for i in range(0,n):
         s_t[i]=sparse_total
     d_s, loc_reqs = sort_drivers(d, sparse, s_t)
     cad = []
     for element in d:
         if element not in d_s:
             cad.append(element)

     for am in cad:
         d_s.append(am)


 for hm in d_s:
  capacity=2
  f1=0

  if driver_i==0:
   target_G = np.load('path to file' +str(up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
   res_G = np.load('path to file' +str(up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
   target_G = np.asarray(target_G, dtype=np.float32)
   target_G = target_G.reshape(576, 576)

   res_G = np.asarray(res_G, dtype=np.float32)
   res_G = res_G.reshape(576, 576)
  else:
   target_G = np.load('use updated path' +str(up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
   res_G = np.load('use updated path' +str(up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
   target_G = np.asarray(target_G, dtype=np.float32)
   target_G = target_G.reshape(576, 576)

   res_G = np.asarray(res_G, dtype=np.float32)
   res_G = res_G.reshape(576, 576)

  elements,loc_sparsity_arr,x,y,xind,yind,loc_arr=loc_sparsity(reqs_matrix)
  increment=1.5
  for i in xind:
      for j in yind:

        res_G[i][j]=res_G[i][j]+increment
  fwait_t=[]

  src=loc[hm]
  path1=[]
  path2=[]
  path12=[]
  path=[]

  if driver_i==(n-1):
   eff_avg=sum(fu_ci)/cd
   wt_avg=sum(fwt)/(n_i*cd+hm)

  dg_a,arr,r,c,dest,capacity,src,path=DAG(res_G,src,capacity)

  driver_i=driver_i+1

  sps=sp(dest)
  dag=np.zeros((abs(r)+1,abs(c)+1))
  k=0
  for i in range(0,abs(r)+1):
   for j in range(0,abs(c)+1):
    dag[i][j]=map[arr[k]]
    k=k+1
  dag=dag.astype(int)
  req_t=np.zeros((abs(r)+1,abs(c)+1))
  vertices=np.zeros((abs(r)+1,abs(c)+1))
  distances=np.zeros((abs(r)+1,abs(c)+1))
  vertices,req_t,distances,capacity=dp(r,c,src,dag,vertices,req_t,distances,capacity)
  pat=ret_path(vertices,src)
  pat.append(src)
  for lko in path:
    path12.append(map[lko])


  ab = itertools.chain(path2[:-1], path12[:-1], pat[::-1])
  res_list=list(ab)
  res_list=res_list[:5]

  for ab in res_list:
    if area_sparse[map1[ab]]==True:
        sparse[hm]=sparse[hm]+1
        sparse_total=sparse_total+1
    all[hm]=all[hm]+1


  if f1==0:
   time_skip=time_gr(loc[hm],map1[res_list[-1]])  #returns time to skip in minutes
   f1=1
  else:
   time_skip=time_gr(loc[hm],res_list[-1])  #returns time to skip in minutes

  time_skip=math.ceil(time_skip/15)  #since time is divided in 15 minute slots
  prev_up[hm]=up[hm]
  up[hm]=map_r[up[hm]]  #give time slot it will give folder name
  up[hm]=up[hm]+time_skip
  up[hm]=map_f[up[hm]]
  loc[hm]=map1[res_list[-1]]  #will next time use their future destination
  o=o+1

  req_path=req_in_path(res_list)

  target_G1=target_G.copy()
  com_reqs=compatible_reqs(res_list,req_path)

  taken_reqs_matrix=loc_sparsity_check(res_list,com_reqs,taken_reqs_matrix,target_G1)



  if com_reqs:

      fwait_t.append(WT(res_list,com_reqs))
      s_m=0
      for i in fwait_t:
        s_m=s_m+i[0]
      avg1=s_m/len(fwait_t)

      fwt[ol]=avg1
      ol=ol+1

      fr[hm]=fr[hm]+fare_cal(com_reqs)

      distan[hm]=distan[hm]+dist_of_path(path,distances,dag,map[dest])
      tim=dist_of_path(path,distances,dag,map[dest])/3
      fu_c=(fare_cal(com_reqs)-dist_of_path(path,distances,dag,map[dest])*(0.1616/1.24))   #1.55 miles can be done in 0.202$
      #u_c is utilitarian utility
      fu_ci[cd]=fu_ci[cd]+fu_c
      futility[hm]=((r_a-1)*futility[hm] + (fu_c/tim))/r_a

  capacity=2
  for vc in com_reqs:

   o_x,o_y=vc
   target_G[o_x][o_y]=0
  np.save('save at different location as the passengers are taken and this data should be updated' +str(up[hm]) + '/res_G.npy', res_G)#.detach().numpy())
  np.save('save at different location as the passengers are taken and this data should be updated' +str(up[hm]) + '/target_G.npy', target_G)#.detach().numpy())



 r_a=r_a+1
 s = np.array(futility)
 sort_index = np.argsort(s)
 d_s=list(sort_index)
 futility=normalize(futility)


 elements,loc_sparsity_arr,x,y,xind,yind,loc_arr=loc_sparsity(reqs_matrix)



loc_r=[]
for item1,item2 in zip(sparse,all):
    if item2!=0:
     loc_r.append(item1/item2)
    else:
        loc_r.append(0)

driver_kl_diver(loc_r,sparse)

lc=[]
for i in loc_sparsity_arr:
    if i>0:
        lc.append(i)

la=[]
for i,j  in zip(x,y):
       la.append(reqs_matrix[i][j])

kl_rider(loc_sparsity_arr,la,np.sum(taken_reqs_matrix))

print("utility per hour",sum(futility)/len(futility))
print("platforms utility",sum(fu_ci)/len(fu_ci))
print("wt",sum(fwt)/len(fwt))

