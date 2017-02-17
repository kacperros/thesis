from utils.RotationAdapter.RotationAdapter import RotationAdapter
from utils.filters.Filter import Filter


class DirectedFilter(RotationAdapter, Filter):
    def __init__(self, img, angle):
        RotationAdapter.__init__(self, img)
        Filter.__init__(self, img)
        self.angle = angle

    def _operate(self):
        self._directed_filter()

# user self._rotated here
    def _directed_filter(self):
        pass

    def _apply_filter(self):
        self.original = self.img
        self.img = self.adapt(self.angle)
