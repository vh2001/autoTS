from pyts.image import MarkovTransitionField


def transform(data):
        mtf = MarkovTransitionField()
        data = mtf.fit_transform(data)
        return data