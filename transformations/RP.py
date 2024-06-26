from pyts.image import RecurrencePlot

def transform(data):
    rp = RecurrencePlot()
    data = rp.fit_transform(data)

    return data