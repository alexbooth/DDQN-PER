class Linear_Anneal():
    def __init__(self, start, end, steps):
        self.value = start
        self.end = end
        self.rate = (start-end) / steps

    def update(self):
        self.value -= self.rate
        if self.value < self.end:
            self.value = self.end
