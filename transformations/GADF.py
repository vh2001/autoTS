from pyts.image import GramianAngularField

def transform(data):

    # perform Gramian Angular Field Transformation
    gadf = GramianAngularField(method='difference')
    data = gadf.fit_transform(data)

    # return transformed data
    return data