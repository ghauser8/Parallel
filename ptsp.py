import multiprocessing
import random
from collections import defaultdict as dd
import math
from itertools import permutations
import sys
import copy
import time

def main(cities, dist):
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
    gincs = {i: multiprocessing.Value('f', fuca([0,i], copy.deepcopy(cities), dist)[1]) for i in ci}
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
    # coerce partial into tuple
    partial = tuple(partial)
    # determine fixed cost of parrtial tour
    pcost = 0
    for c in range(len(partial)):
        if c == len(partial) - 1:
            break
        pcost += dist[partial[c]][partial[c+1]]
    # figure out which cities are still unvisited
    unseen = [i for i in range(max_cities) if i not in partial]
    size = math.factorial(len(unseen))
    if len(unseen) == 0:
        with global_incumb.get_lock():
            for i in range(len(partial)):
                global_incumb[i] = partial[i]
        gic.value = pcost
        return (partial, pcost)

    if len(partial) > 0:
        start = partial[-1]
    else:
        start = unseen[0]
    
    incumbent = None
    inCost = math.inf
    if log:
        print(f's{pid} - Starting search over {size} permutations')

    pcount = 0
    for p in permutations(unseen):
        # compute cost of ptour
        if len(partial) > 0:
            sp = (start, *p)   
        else:
            sp = p
        spcost = 0
        for c in range(len(sp)):
            if c == len(sp)-1:
                break
            spcost += dist[sp[c]][sp[c+1]]

        logged = False
        pcount += 1
        if spcost + pcost < inCost:
            inCost = spcost + pcost
            incumbent = sp
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
    if len(partial) > 0:
        if minned and gic is not None:
            with global_incumb.get_lock():
                incumb = partial + incumbent[1:]
                for c in range(len(incumb)):
                    global_incumb[c] = incumb[c]
        # global_incumb = (*partial, incumbent[1:])
        return ((*partial, *incumbent[1:]), inCost)
    else:
        if minned and gic is not None:
            with global_incumb.get_lock():
                for c in range(len(incumbent)):
                    global_incumb[c] = incumbent[c]
        return (incumbent, inCost)


def cuca(city, cities, dist):
    '''
    DESCRIPTION: closest unvisited city algorithm
    city (int): city to start from (must be a member of cities)
    cities (list): list of cities (each city is an integer)
    dist (dict of dicts): for city key k, get distances dict
    '''
    # so we don't accidentally make a sub-tour
    cities.remove(city)
    
    tour = [city]
    cost = 0
    # loop until we've built a complete tour
    while len(cities) > 0:
        nxt = min([(dist[tour[-1]][k],k) for k in cities], key = lambda x: x[0])
        cost += nxt[0]
        tour.append(nxt[1])
        cities.remove(nxt[1])
    
    return (tour, cost)

def fuca(partial, cities, dist):
    '''
    DESCRIPTION: furthest unvisited city algorithm
    city (int): city to start from (must be a member of cities)
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
        nxt = max([(dist[tour[-1]][k],k) for k in cities], key = lambda x: x[0])
        cost += nxt[0]
        tour.append(nxt[1])
        cities.remove(nxt[1])
    
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
    try:
        n = int(n)
    except:
        print('Pls pass an integer argument for number of cities')
        raise ValueError

    cities, dist = instGen(n)
    main(cities, dist)

    # gstart = time.time()
    # incmb, cst = localSearch(len(cities), dist, partial = [0], log=True)
    # gend = time.time()
    # print(f'Global serial search culminated in {gend - gstart} seconds')
    # print(f'{incmb}')
    # print(f'{cst}')

