import multiprocessing
import random
from collections import defaultdict as dd
import math
from itertools import permutations
from itertools import combinations
import sys
import copy
import time
import os
import queue as que

def CPsearch(cities, dist, nsize):
    '''
    DESCRIPTION: use multiprocessing to search for lowest cost tour of cities.
        Chunks and queues search neighborhoods; global incumbent used to fathom
        queued chunks before/while processing them.
    '''
    print('Building queue.')
    qstart = time.time()
    # list of cities minus start city (0)
    pcities = [c for c in cities if c != 0]

    # use cuca to develop incumbent solution
    global_incumbent_cost = multiprocessing.Value('f', cfuca([0], copy.deepcopy(cities), dist)[1])
    global_incumbent = multiprocessing.Array('i',len(cities))
    cinc = math.inf
    cint = None
    for c in pcities:
        t, c = cfuca([0,c],copy.deepcopy(cities),dist, minimize = True)
        if c < cinc:
            cinc = c
            cint = t
    global_incumbent_cost.value = c
    for c in range(len(cint)):
        global_incumbent[c] = cint[c]

    print(f'Cuca produced incumbent tour of {[c for c in global_incumbent]}')
    print(f'Starting incumbent cost: {global_incumbent_cost.value}')
    # queue neighborhoods to search based on user defined chunk size.
    plen = len(cities) - nsize
    
    queue = multiprocessing.Queue()
    
    tchunks = (math.factorial(len(pcities))/math.factorial(len(pcities) - (plen)))
    print(f'{tchunks} total search neighborhoods before fathoming')
    sns = 0
    for pt in permutations(pcities, plen):
        # check that cost of partial tour isn't greater than current incumbent
        # zpt = (0, *pt)
        if tourCost(pt, dist) < global_incumbent_cost.value:
            queue.put_nowait(pt)
            sns += 1
    
    print(f'Fathomed {tchunks - sns} neighborhoods; {sns} left to search')
    qend = time.time()
    print(f'Queue took {qend - qstart:.2f} seconds to build ({(qend-qstart)/60:.2f} minutes).')

    # spin up max number of processes to start consuming neighborhoods from queue
    pn = os.cpu_count()
    pn = math.floor(pn*0.75) # leave at least a quarter of available processors alone
    procs = [
        multiprocessing.Process(target=qfedlocalSearch, 
            args = (queue, cities, dist),
            kwargs={'finter': 0.5,
                    'git': global_incumbent,
                    'gic': global_incumbent_cost,
                    'log': True,
                    'pid': i}
        )
        for i in range(pn)
    ]

    print(f'Starting {pn} searches')
    sstart = time.time()
    # start procs
    for p in procs:
        p.start()
    
    # await procs
    for p in procs:
        p.join()
    
    send = time.time()
    print(f'Optimal tour: {[c for c in global_incumbent]}')
    print(f'Optimal tour cost: {global_incumbent_cost.value}')
    print(f'Search time: {send - sstart:.2f} seconds ({(send - sstart)/60:.2f} minutes)')
    print(f'Total run time: {(qend - qstart) + (send - sstart):.2f} seconds ({((qend - qstart) + (send - sstart))/60:.2f} minutes)')
    
    # attempt to dequeue so that main will terminate
    latest = None
    while True:
        try:
            latest = queue.get_nowait() 
        except:
            break

    
    return


    

def qfedlocalSearch(queue, 
                    cities, 
                    dist,
                    finter = 0.25, 
                    git = None, 
                    gic = None, 
                    log = False, 
                    pid = 0):
    '''
    DESCRIPTION: read partial tours from shared queue and search local neighborhood
    '''
    # attempt to read queue so long as latest partial off the queue can't be fathomed
    incumbent_tour = [i for i in cities]
    
    while True:
        try:
            pt = queue.get(timeout=0.05)
        except que.Empty:
            # an empty queue means all neighborhoods have been searched
            break
        
        # check that current neighborhood can't be fathomed
        ptcost = tourCost(pt, dist)
        if ptcost > gic.value:
            # if log:
            #     print(f's{pid} - I fathomed the {pt} branch!!') 
            continue

        # begin processing current neighborhood
        unseen = [i for i in cities if i not in pt]
        start = pt[-1]
        incumbent_cost = gic.value
        with git.get_lock():
            for c in range(len(git)):
                incumbent_tour[c] = git[c]
        
        size = math.factorial(len(unseen))
        # if log:
        #     print(f's{pid} - Starting search over {size} permutations')
        
        pcount = 0
        for p in permutations(unseen):
            # sp = (start, *p)
            # spc = tourCost(sp, dist)
            # tcost = spc + ptcost + dist[sp[-1]][0] # I'm not sure 0 is correct here
            tour = pt + p
            tcost = tourCost(tour, dist, partial = False)
            pcount += 1
            # update local incumbents
            if tcost < incumbent_cost:
                incumbent_cost = tcost
                incumbent_tour = list(tour)
            
            # compare against global incumbent if we've gotten far enough
            if (pcount/size) % finter == 0:
                with gic.get_lock():
                    if incumbent_cost < gic.value:
                        gic.value = incumbent_cost
                        with git.get_lock():
                            for c in range(len(incumbent_tour)):
                                git[c] = incumbent_tour[c]
                            
                    elif ptcost > gic.value:
                        # break out of current neighborhood search
                        if log:
                            print(f's{pid} - I fathomed the {pt} branch after {pcount/size:.0%} of local search!')
                        break
    return

    
    

def tourCost(tour, dist, partial = True):
    '''
    DESCRIPTION: calculate cost of (partial) tour
    tour (iterable): (partial) tour to evaluate
    dist (dict of dicts): distance matrix
    partial (bool): whether or not the tour is partial
    '''
    cost = 0
    for c in range(len(tour)):
        if c == len(tour) - 1:
            break
        cost += dist[tour[c]][tour[c+1]]
    
    if not partial:
        cost += dist[tour[-1]][tour[0]]
    
    return cost





def NFPsearch(cities, dist):
    '''
    DESCRIPTION: use multiprocessing to search for lowest cost tour of cities
    dist (dict of dicts): for city key k, get distances dict 
    cities (list): list of cities (each city an integer)
    '''
    '''
    main will create and coordinate child processes, maintain shared data, &
        return result
    '''

    '''
    First approach: non-fathoming parallel searches along unique starting cities.
        - Current (25 NOV) approach does not search entire space; will only search
            first min(pn,len(cities)) - 1 neighborhoods 
                (permutations from top-level branches).
        - Next step is to build a pool of processors and queue searches so that 
            all neighborhoods (top level branches) are searched.
    '''
    start = time.time()
    # figure out how many processes to produce (hard code to 9 for now)
    #   will actually get at most pn - 1 searches; see ci
    pn = 19

    # assume WLOG we always start at city 0
    ci = [i for i in range(min(pn,len(cities))) if i != 0]

    # define global incumbent
    gincs = {i: multiprocessing.Value('f', cfuca([0,i], copy.deepcopy(cities), dist)[1]) for i in ci}
    gints = {i: multiprocessing.Array('i', len(cities)) for i in ci}

    # define process(es)
    # each process is a local search starting at city 0
    processes = [multiprocessing.Process(
        target=localSearch, 
        args=(len(cities), dist), 
        kwargs={'partial': [0,i], 
            'pid':i, 
            'global_incumb':gints[i], 
            'gic':gincs[i], 
            'log': True
        }) 
        for i in ci
    ]
    
    
    # start processes
    print(f'Starting {len(processes)} searches')
    for p in processes:
        p.start()

    # wait for process to finish
    print('awaiting searches')
    for p in processes:
        p.join()

    # make results visible
    res = {i:gincs[i].value for i in gincs}
    mini = pn + 1
    mincost = math.inf
    for i in gincs:
        if res[i] < mincost:
            mincost = res[i]
            mini = i
    end = time.time()
    print(f'Optimal tour: {[c for c in gints[mini]]}')
    print(f'Optimal tour cost: {gincs[mini].value}')
    print(f'Searches culminated in {(end - start)} seconds ({(end - start)/60} minutes)')




def localSearch(max_cities, 
                dist, 
                partial = [], 
                log = False, 
                pid = 0, 
                global_incumb = None, 
                gic = None):
    '''
    DESCRIPTION: build the lowest cost tour by brute force given a partial tour
    max_cities (int): max index of list of cities
    dist (dict of dicts): distance matrix
    partial (list): ordered list of city indicies representing a partial tour
    '''
    
    # determine fixed cost of partial tour
    pcost = tourCost(tour = partial, dist = dist)
    # figure out which cities are still unvisited
    unseen = [i for i in range(max_cities) if i not in partial]
    size = math.factorial(len(unseen))
    if len(unseen) == 0:
        if gic is not None:
            with global_incumb.get_lock():
                for i in range(len(partial)):
                    global_incumb[i] = partial[i]
            gic.value = pcost
        return (partial, pcost)
    
    incumbent = None
    inCost = math.inf
    if log:
        print(f's{pid} - Starting search over {size} permutations')

    pcount = 0
    for p in permutations(unseen):
        # compute cost of ptour
        if len(partial) > 0:   
            intour = partial + list(p)
        else:
            intour = list(p)
        spcost = tourCost(tour = intour, dist = dist, partial = False)

        logged = False
        pcount += 1
        if spcost < inCost:
            inCost = spcost 
            incumbent = intour
            if log:
                print(f's{pid} - New incumbent cost: {inCost:.2f}; {pcount/size:.2%} search complete')
                logged = True
        if log and not logged and ((pcount/size)*100) % 5 == 0:
            print(f's{pid} - {pcount/size:.0%} search complete')
    
    minned = False
    if gic is not None:
        with gic.get_lock():
            if gic.value > inCost:
                gic.value = inCost
                minned = True
    
    if minned and gic is not None:
        with global_incumb.get_lock():
            for c in range(len(incumbent)):
                global_incumb[c] = incumbent[c]

    return (incumbent, inCost)
    


def cfuca(partial, cities, dist, minimize = False):
    '''
    DESCRIPTION: furthest unvisited city algorithm
    partial (list of ints): partial tour to start from
    cities (list): list of cities (each city is an integer)
    dist (dict of dicts): for city key k, get distances dict
    '''
    # so we don't accidentally make a sub-tour
    for c in partial:
        cities.remove(c) 
    
    tour = copy.deepcopy(partial)
    cost = 0
    for c in range(len(tour)):
        if c == len(tour)-1:
                break
        cost += dist[tour[c]][tour[c+1]]

    # loop until we've built a complete tour
    while len(cities) > 0:
        if minimize:
            nxt = min([(dist[tour[-1]][k],k) for k in cities], key = lambda x: x[0])
        else:
            nxt = max([(dist[tour[-1]][k],k) for k in cities], key = lambda x: x[0])
        
        cost += nxt[0]
        tour.append(nxt[1])
        cities.remove(nxt[1])

    cost += dist[tour[-1]][tour[0]]
    
    return (tour, cost)

def instGen(n = 10):
    '''
    DESCRIPTION: generate a random TSP problem for n cities
    n (int): number of cities
    '''

    # put cities randomly on xy grid
    coords = [(random.randint(1,5000),random.randint(1,5000)) for i in range(n)]

    dist = {i:dd(float) for i in range(n)}
    # generate distances
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist[i][j] = math.sqrt(((coords[i][0] - coords[j][0])**2 + \
                (coords[i][1] - coords[j][1])**2))

    # return cities list and dists
    return ([i for i in range(n)], dist)


def test():
    cities, dist = instGen(10)
    return (cuca(0, cities, dist), dist)

if __name__ == '__main__':
    n = sys.argv[1]
    m = sys.argv[2]
    try:
        n = int(n)
    except:
        print('Pls pass an integer argument for number of cities')
        raise ValueError

    try:
        m = int(m)
    except:
        print('Pls pass an integer argument for size of subpermutations')
        raise ValueError

    cities, dist = instGen(n)
    CPsearch(cities, dist, m)

    gstart = time.time()
    incmb, cst = localSearch(len(cities), dist, partial = [0], log=True)
    gend = time.time()
    print(f'Global serial search culminated in {gend - gstart} seconds')
    print(f'{incmb}')
    print(f'{cst}')

