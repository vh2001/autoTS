from sklearn.metrics import classification_report
import pandas as pd
import importlib


# # custom packages
# from autoTS.models.wrapper.VGG16_wrapper import VGG16_wrapper
# from autoTS.models.wrapper.InceptionTime_wrapper import InceptionTime_wrapper

import src.config as cfg

# add your model wrappers here
model_wrappers_imports = {
    "VGG16": "autoTS.models.wrapper.VGG16_wrapper",
    "InceptionTime": "autoTS.models.wrapper.InceptionTime_wrapper"
}



def run_model(data: list):

    model_wrapper_module = importlib.import_module(model_wrappers_imports[cfg.MODEL])
    model_wrapper = getattr(model_wrapper_module, f"{cfg.MODEL}_wrapper")
    
    train_loader, test_loader = model_wrapper.train_test_data(data, cfg.BATCH_SIZE, cfg.SHUFFLE, cfg.DATA_SPLIT, cfg.FOLDS)
    
    model_wrapper.train(train_loader, cfg.EPOCHS, cfg.LR, cfg.CALLBACKS)

    y_pred, y_true = model_wrapper.predict(test_loader)

    # save model 
    model_wrapper.save_model(cfg.SAVE_PATH)

    if cfg.TASK == "classification":
        cr = classification_report(y_pred, y_true, output_dict=True)
        df = pd.DataFrame(cr)

        df.to_csv(f"{cfg.SAVE_PATH}/classification_report.csv")
    
    else:
        raise NotImplementedError("Only classification is supported for now")



    #
