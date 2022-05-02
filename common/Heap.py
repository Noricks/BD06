from common.Verbose import Verbose


class Heap(Verbose):

    def __init__(self, verbose=True):

        self.heap = []  #
        self.handle_dict = {}  #
        super(Heap, self).__init__(verbose)

    def _siftup(self, pos):
        end_pos = len(self.heap)
        lchild = 2 * pos + 1
        rchild = lchild + 1
        min_pos = lchild
        while min_pos < end_pos:
            if self.heap[min_pos] > self.heap[pos]:
                min_pos = pos
            if rchild < end_pos and self.heap[min_pos] > self.heap[rchild]:
                min_pos = rchild
            if min_pos != pos:
                self.printer("exchange position{}:{}and{}:{}".format(
                    min_pos, self.heap[min_pos], pos, self.heap[pos]))
                self.printer("exchange position{}and{}".format(min_pos, pos))
                self.heap[min_pos], self.heap[pos] = self.heap[pos], self.heap[min_pos]
                self.printer("update handle_dict")
                self.printer("{}->{}".format(self.heap[pos][1], min_pos))
                self.printer("{}->{}".format(self.heap[min_pos][1], pos))
                self.handle_dict[self.heap[pos][1]] = pos
                self.handle_dict[self.heap[min_pos][1]] = min_pos
                pos = min_pos
                lchild = 2 * pos + 1
                rchild = lchild + 1
                min_pos = lchild
            else:
                break

    def _siftdown(self, pos):
        new_item = self.heap[pos]
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent_item = self.heap[parentpos]
            if new_item < parent_item:
                self.heap[pos] = parent_item
                self.handle_dict[parent_item[1]] = pos
                pos = parentpos
                continue
            break
        self.heap[pos] = new_item
        self.handle_dict[new_item[1]] = pos

    def heapify(self, x):

        n = len(x)
        self.heap = x
        for i in range(n):
            self.handle_dict[x[i][1]] = i

        for i in reversed(range(n // 2)):
            self._siftup(i)

    def push(self, data):
        key, handle = data
        try:
            pos = self.handle_dict[handle]
            if self.heap[pos][0] > key:
                self.decrease_key(data)
            elif self.heap[pos][0] < key:
                self.increase_key(data)
        except:
            self.heap.append(data)
            self.handle_dict[handle] = len(self.heap) - 1
            self._siftdown(len(self.heap) - 1)

    def decrease_key(self, data):

        new_key, handle = data
        pos = self.handle_dict[handle]
        if self.heap[pos][0] < new_key:
            raise ValueError("new key is larger than the origin key")
        self.heap[pos][0] = new_key
        self._siftdown(pos)

    def increase_key(self, data):

        new_key, handle = data
        pos = self.handle_dict[handle]
        if self.heap[pos][0] > new_key:
            raise ValueError("new key is smaller than the origin key")
        self.heap[pos][0] = new_key
        self._siftup(pos)

    def pop(self):

        last_item = self.heap.pop()
        if self.heap:
            return_item = self.heap[0]
            self.heap[0] = last_item

            self.handle_dict[last_item[1]] = 0
            del self.handle_dict[return_item[1]]
            self._siftup(0)
        else:
            return_item = last_item
            del self.handle_dict[return_item[1]]
        return return_item

    def min(self):
        return self.heap[0]

    @property
    def is_empty(self):
        return True if len(self.heap) == 0 else False
