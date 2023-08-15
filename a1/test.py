import heapq


def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return h #[heapq.heappop(h) for i in range(len(h))]


x=heapsort([1, 1, 5, 7, 9, 2, 4, 6, 8, 0])
if 70 in x:
    print(x)
while x :
    print(x)
    x.heappop()
