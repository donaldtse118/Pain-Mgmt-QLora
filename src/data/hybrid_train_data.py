# Hybrid train data is to solve original medicial dataset imbalance class problem

# 3rd parties
from datasets import DatasetDict, Dataset
import pandas as pd

# local import
from utils.logger import get_logger

logger = get_logger(__name__)


exected_dosage = ['high', 'low', 'omitted']


def get_hybrid_train_data(df_orig: pd.DataFrame, 
                          df_aug: pd.DataFrame,
                          target_size=150,
                          persist_path: str = None
                          ):
    """
    Base on original medicial dataset, in each iteration supplement the least occur pain_type and dosage sample from augmented data
    (performance fine for small dataset)
    """

    # df_orig = load_from_disk("local/data/processed")["train"].to_pandas()
    # df_aug = load_from_disk(
    #     "local/data/augmented_extend_pain_type_desc")["train"].to_pandas()

    df_orig['data_source'] = 'orig'
    df_aug['data_source'] = 'augmented'

    df_orig['dosage'] = df_orig['dosage'].apply(
        lambda x: x.split(" ")[0].lower())
    df_aug['dosage'] = df_aug['dosage'].apply(
        lambda x: x.split(" ")[0].lower())

    df_aug['pain_type'] = df_aug['pain_type'].str.replace("_", " ")

    df_result = df_orig.copy()

    step = 1

    while df_result.shape[0] < target_size:
        # find out least record

        dosage_hist = df_result.dosage.value_counts()

        missed_dosage = [d for d in exected_dosage if d not in dosage_hist]
        if len(missed_dosage) > 0:
            least_dosage = missed_dosage[0]
        else:
            least_dosage = dosage_hist.idxmin()
        least_pain_type = df_result.pain_type.value_counts().idxmin()

        # select from augement data
        df_sel = df_aug[(df_aug.pain_type == least_pain_type)
                        & (df_aug.dosage == least_dosage)].head(step)
        df_aug.drop(df_sel.index, inplace=True)

        df_result = pd.concat([df_sel, df_result], ignore_index=True)
    
    logger.info(f"hybrid data distribtuion, ttl : {df_result.shape[0]}")
    logger.info(df_result.groupby(["dosage", "data_source"]).size())

    # ttl : 150
    # dosage   data_source
    # high     augmented      50
    # low      augmented      19
    #          orig           31
    # omitted  augmented      46
    #          orig            4

    splits = DatasetDict({
        'train': Dataset.from_pandas(df_result),
    })
    
    # splits.save_to_disk("local/data/hybrid_extended_pain_type_desc")
    if persist_path:
        splits.save_to_disk(persist_path)




    return splits



