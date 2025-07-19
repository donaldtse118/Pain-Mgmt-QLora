# local import
from data import preprocess, argumentation, hybrid_train_data
from model import fine_tune_model as ft
from model import evaluation
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

dataset_medical = preprocess.get_medical_dataset()
logger.info(f"## Prepared medical data:")
for split_name, split_dataset in dataset_medical.items():
    logger.info(f"{split_name}: {len(split_dataset)} samples")

dataset_augmented = argumentation.get_augmented_dataset()
logger.info(f"## Prepared augmented data:")
for split_name, split_dataset in dataset_augmented.items():
    logger.info(f"{split_name}: {len(split_dataset)} samples")

logger.info(f"## Prepaing hybrid train data:")
dataset_hybrid = hybrid_train_data.get_hybrid_train_data(df_orig=dataset_medical['train'].to_pandas(),
                                        df_aug=dataset_augmented['train'].to_pandas(),
                                        target_size=150
                                        )
logger.info(f"## Prepared hybrid train data:")
for split_name, split_dataset in dataset_hybrid.items():
    logger.info(f"{split_name}: {len(split_dataset)} samples")

logger.info(f"# Start to fine tune model")
_, _, ft_model_path = ft.fune_tune_model(dataset=dataset_hybrid['train'])
logger.info(f"Finished to fine tune model")


logger.info(f"Start evaluation...")
test_datasets = {
    "augmented": dataset_augmented["test"].select(range(20)),
    "medicial": dataset_medical["test"].select(range(20)),
}

models = {
    "baseline": config.PRETRAIN_MODEL_NAME,
    "finetune": ft_model_path,
}

evaluation.evaluate(models=models,
                    test_datasets=test_datasets)
logger.info(f"Finished Evaluation")

