class Filter:
    def __init__(self, img):
        self.img = img

    def filter(self):
        self._pre_process()
        self._apply_filter()
        return self._post_process()

    def _pre_process(self):
        pass

    def _post_process(self):
        return self.img

    def _apply_filter(self):
        pass
