class UnionFind(object):
    def __init__(self, n=0):
        self.N = n
        self.uf = [-1 for i in range(n + 1)]
        self.sets_count = n

    def append(self, n=1):
        self.N += n
        self.uf += [-1 for i in range(n)]
        self.sets_count += n

    def _find(self, p):
        if self.uf[p] < 0:
            return p
        self.uf[p] = self._find(self.uf[p])
        return self.uf[p]

    def find(self,p):
        if self.uf[p+1] < 0:
            return p+1
        return self.uf[p+1]

    def union(self, p, q):
        proot = self._find(p+1)
        qroot = self._find(q+1)
        if proot == qroot:
            return
        elif self.uf[proot] > self.uf[qroot]:
            self.uf[qroot] += self.uf[proot]
            self.uf[proot] = qroot
        else:
            self.uf[proot] += self.uf[qroot]
            self.uf[qroot] = proot
        self.sets_count -= 1

    def is_connected(self, p, q):
        return self._find(p+1) == self._find(q+1)

    def update(self):
        for p in range(self.N):
            f = self._find(p+1)
        return
