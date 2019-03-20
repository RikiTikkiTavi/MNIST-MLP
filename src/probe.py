class Probe:
    def __init__(self):
        self.d = {1: 1}

    def do(self):
        self.d[1] += 1


p = Probe()
p.do()
print(p.d)
