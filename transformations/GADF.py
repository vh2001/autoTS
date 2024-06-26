from pyts.image import GramianAngularField

def transform(data):
    gadf = GramianAngularField(method='difference')
    data = gadf.fit_transform(data)

    return data