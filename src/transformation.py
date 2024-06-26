import importlib
def transform(data, transformation):
    # check if we use transformation if not return data
    if transformation == "None":
        return data
    # import transformation module
    transformation_module = importlib.import_module(f"autoTS.transformations.{transformation}")

    # get transformation function
    transformation_function = getattr(transformation_module, transformation)

    # apply transformation
    data = transformation_function(data)

    return data