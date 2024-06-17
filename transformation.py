

def transform(data, transformation):

    if transformation == "GADF":
        from pyts.image import GramianAngularField
        gadf = GramianAngularField(method='difference')
        data = gadf.fit_transform(data)
    elif transformation == "RP":
        from pyts.image import RecurrencePlot
        rp = RecurrencePlot()
        data = rp.fit_transform(data)
    else:
        raise ValueError("Invalid transformation")

    return data