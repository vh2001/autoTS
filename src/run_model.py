from sklearn.metrics import classification_report
import pandas as pd



# custom packages
from models.VGG16 import VGG16_wrapper
import src.config as cfg
model_wrappers = {
    "vgg16": VGG16_wrapper
}

# models = {
#     "vgg16": models.vgg16(pretrained=False)

# }


def run_model(data: list):


    model_wrapper = model_wrappers[cfg.MODEL]()
    
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
