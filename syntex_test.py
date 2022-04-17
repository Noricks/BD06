# %%
class A:
    def __init__(self, q):
        self.q = q

    def p(self):
        print(self.q)

class B(A):
    pass