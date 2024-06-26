from pyts.image import GramianAngularField


def transform(data):
        gasf = GramianAngularField(method='summation')
        data = gasf.fit_transform(data)

        return data