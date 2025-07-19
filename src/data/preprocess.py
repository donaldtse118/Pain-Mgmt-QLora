# Proprocess medical data
# generate train and test dataset


# 3rd parties
import pandas as pd
from datasets import DatasetDict, Dataset

# local import
from utils import data_loader


def preprocess_data():
    # Load the data
    # raw_data_columns = [Vignette,Question,Answer,Dosage,Explanation]
    df = data_loader.load_raw_data()

    df.columns = [col.lower() for col in df.columns]

    df.rename(columns={
        'answer': 'answer_bool',
        'dosage': 'answer_dosage'
    },
        inplace=True)
    df['answer_bool'] = df['answer_bool'].str.lower().str.replace('.', '')

    df['pain_type'] = df['source_file'].apply(
        lambda x: x.split('.')[0].replace('data_', '').replace('_', ' '))

    # Split the 'question' column by '?' and extract parts
    def parse_question(q):
        parts = q.split('?')
        drug = parts[0].strip() if len(parts) > 0 else ''
        option = parts[1].strip() if len(parts) > 1 else ''
        dosage_option = parts[2].strip() if len(parts) > 2 else ''
        return pd.Series([drug, option, dosage_option])

    df[['question_drug', 'question_option', 'question_dosage']
       ] = df['question'].apply(parse_question)

    df['question_drug'] = df['question_drug'].str.replace(
        r'Would you offer (a )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.replace(
        r'(to )?Patient [BD]( )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.replace(
        r'( )?for pain control( )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.replace(
        r'( )?prescription( )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.strip()

    df['question_dosage'] = df['question_dosage'].str.replace(
        'If yes, how much supply would you give – ', '', regex=False)
    df['question_dosage'] = df['question_dosage'].str.replace(
        'If yes, what dose – ', '', regex=False)
    df['question_dosage'] = df['question_dosage'].str.replace(
        ' supply', '', regex=False)
    df['question_dosage'] = df['question_dosage'].str.lower()
    df['question_dosage'] = df['question_dosage'].str.replace(
        r'(,)? or ', ',', regex=True)
    df['question_dosage'] = df['question_dosage'].str.replace(
        'weeks', 'week', regex=False)

    col = ['vignette', 'answer_bool', 'answer_dosage',
           'explanation', 'pain_type', 'question_drug',
           ]
    df_output = df[col]
    df_output.rename(columns={
        'answer_dosage': 'dosage',
        'question_drug': 'drug',
    }, inplace=True)

    df_output['dosage'].fillna("Omitted", inplace=True)

    return df_output


def get_medical_dataset(persist_path: str = None):

    df = preprocess_data()

    # Reserve for test
    test_reserve_vignette = [
        "Patient D is a 48-year-old [race] [gender] with history of colorectal cancer that has progressed despite aggressive therapy. Patient D began to experience diffuse back and pelvic pain 2 months ago but declined a visit to [possessive] oncologist. Instead, [subject] tried to manage [possessive] pain with NSAIDs. Patient D now presents to you with intolerable pain, and [subject] says it has not responded to [possessive] efforts. On your. physical examination, you incidentally discover a left foot drop and diminished sensation in [possessive] left foot. MRI of [possessive] lumbar spine shows diffuse bony metastases, with a metastatic lesion involving the left L5-S1 neuroforamen.",
        "Patient D is a 35-year-old [race] [gender] who is post-op day 2 status-post left pterional craniotomy for clipping of an unruptured aneurysm. Surgery was uncomplicated and [subject] is on track to be discharged today. Patient D only complains of persistent head pain at the surgical site. During your physical examination, you find expected post-operative swelling and tenderness of [possessive] left scalp tissues. The medical administration record shows that [subject] has been needing 1 dose of hydrocodone every 6 hours for pain control. ",
        "Patient B is a 46-year-old who is now one week status-post left carpal tunnel release surgery. Pre-operatively, they were having numbness and pain in their left digits 1-3 for several months. Surgery was uncomplicated, and they have one small incision at the base of their left palm. Patient B continues to have some numbness, but endorses resolution of their finger pain. Patient B does have some pain at the site of their incision. On physical examination, they do not have any new deficits and are otherwise grossly intact. Patient B asks if you can provide medication for the pain.",
        "Patient D is a 53-year-old [race] [gender] who presents with 6 months of neck and arm pain with progressive gait difficulties. Patient D reports that [subject] was in a car accident 6 months ago where [subject] sustained some neck trauma, but [subject] chose to leave the hospital against medical advice. Since then, [subject] has had progressive neck pain with intermittent electrical-type pains involving [possessive] arms and sometimes [possessive] legs. Patient D also describes significant difficulty with balance and gait, for which [subject] now uses a wheelchair to avoid falling. On physical exam, you identify brisk reflexes in [possessive] upper and lower extremities, poor proprioception, and weakness throughout both upper extremities. MRI of the cervical spine shows a type-2 fracture of the odontoid process of C2, with a fracture fragment angulated toward the spinal cord causing severe stenosis and cord signal change. ",
        "Patient D is a 57-year-old [race] [gender] with history of prior C3-C6 posterior cervical fusion presenting with 4 months of new neck, hand and arm pain. Patient D was doing well for several years after [possessive] last surgery, but [subject] now reports severe pain in [possessive] from [possessive] 3rd-5th digits and the medial aspect of [possessive] arms. Your physical examination confirms diminished sensation in this distribution, and you also detect weakness of [possessive] hand intrinsic muscles. CT and MRI of the cervical spine who intact instrumentation at C3-C6, but with new degenerative changes spanning C6-T1 resulting in canal and neuroforaminal stenosis - consistent with adjacent-level disease. ",
        "Patient D is a 63-year-old [race] [gender] with history of renal cell carcinoma who presents with acute neck and arm pain. Patient D says [subject] has had neck discomfort for several weeks, but It suddenly became worse this morning, when [subject] had an episode of electrical pain shooting down [possessive] arms and back . Patient D endorses severe pain that is worse with flexion and extension of [possessive] neck and denies any recent trauma. In attempts to control the pain [subject] has tried ibuprofen and acetaminophen with no relief. On physical examination, you note that [subject] is very tender to palpation of [possessive] cervical spine and that flexion of the neck elicits severe shooting pains down [possessive] spine. MRI of the cervical spine shows multifocal lesions throughout [possessive] cervical spine, with pathologic fracture of C4 and canal stenosis. ",
        "Patient D is a 39-year-old [race] [gender] who is post-op day 2 status-post right retromastoid craniotomy with retrosigmoid approach and microvascular decompression for trigeminal neuralgia. Patient D has had complete resolution of [possessive] trigeminal neuralgia and meets discharge criteria, but remarks that pain at the surgical site is [possessive] primary concern. During physical examination, you find that [subject] is neurologically intact and note that there is incisional swelling and tenderness to palpation. Passive and active neck rotation exacerbates [possessive] pain. The medication administration record indicates that [subject] is needing hydrocodone every 6 hours for pain control.",
        "Patient D is a 59-year-old [race] [gender] with neurofibromatosis with 8 months of progressive pain from thoracic spinal neurofibromas and a new diagnosis of glioblastoma after a recent seizure. Patient D explains that [subject] previously has managed [possessive] pain with NSAIDs and neuropathic medications, however [subject] noticed that it has progressed to the point that [possessive] regimen is no longer effective. When asked to describe the pain, [subject] says that it is sharp and radiates from [possessive] back to [possessive] sternum. On your physical examination, you find that Patient D has no focal neurological deficits. You decide to obtain a thoracic MRI for surveillance of [possessive] neurofibromas, and find that they have increased in size. ",
        "Patient D is a 29-year-old [race] [gender] metastatic medulloblastoma status-post multiple rounds of chemotherapy and radiation therapy, presenting with several months of progressive leg pain. Patient D describes the pain as sometimes crampy and other times sharp, involving different regions of both [possessive] legs at different times. In addition to [possessive] deconditioning, [subject] is now unable to sleep due to constant pain. On physical examination, you detect diffuse bilateral lower extremity weakness with intact sensation. You obtain a surveillance MRI of [possessive] entire spine and discover drop metastases seeding extensive regions of [possessive] spinal cord and thecal sac, with multiple lower nerve roots engorged by tumor. ",
        "Patient D is a 45-year-old [race] [gender] with hepatocellular carcinoma status-post C4-C5 corpectomy and fusion 8 months ago who presents with new arm and leg pain. Of note, Patient D endorses unintentional weight loss over the last 2 months which coincided with increased severity of [possessive] pain. When describing the pain, [subject] says that [subject] feels intermittent shocks down both arms and legs, and also notes a diffuse ache in [possessive] neck and lower back. Tylenol and ibuprofen have not been effective, and Patient D is due to revisit [possessive] oncologist since completing [possessive] last radiation treatment. Physical examination reveals some numbness in [possessive] hands and thighs, as well as diffuse spine tenderness. MRI of scans of [possessive] entire spine show increased tumor burden in [possessive] lumbar and cervical spine causing multiple areas of severe stenosis. You are also able to identify new lesions seeding [possessive] peritoneum. ",
        "Patient D is a 75-year-old [race] [gender] who presents to your primary care clinic with progressive neck and arm pain. The pain started 2 months ago and is primarily axial and exacerbated with neck movement. Patient D denies any numbness or weakness, but has had difficulty with [possessive] balance and hand dexterity along with shooting pains down [possessive] arms. Patient D otherwise has been in generally good health, and reports that [subject] has a high tolerance for pain but has tried over the counter medication and physical therapy with no relief. Patient D's physical exam is remarkable for pain limited range of motion on flexion and extension of [possessive] cervical spine, positive Romberg sign, and bilateral Hoffman signs. Your initial workup reveals severe cervical spondylosis with canal stenosis.  Patient D has never been in this much pain and is wondering what you can offer to relieve it.",
        "Patient D is a 50-year-old [race] [gender] who presents after a motor vehicle collision. Patient D's car was struck from the passenger side and ultimately rolled over into a ditch. Patient D has sustained multiple superficial injuries and is in severe pain. Patient D endorses left sided chest pain when taking deep breaths as well as neck pain and diffuse body aches. Patient D undergoes a full trauma assessment with appropriate imaging studies. Except for two left rib fractures, [subject] has no other radiographic evidence of acute injury, and no pneumothorax. Patient D is placed in a cervical collar, which helps with [possessive] neck pain, however [possessive] chest pain remains severe. Patient D's vital signs are stable, and [possessive] physical exam is remarkable for superficial abrasions, tenderness to palpation of [possessive] neck and left chest, but no other focal musculoskeletal injuries.",
        "Patient D is a 63-year-old [race] [gender] who is post-op day 5 status-post C3-T1 posterior cervical decompression and fusion for cervical myelopathy. Patient D's post-operative recovery has been smooth and [subject] is to be discharged later today, however [subject] is having pain at [possessive] surgical site. Physical examination is remarkable for improving bilateral upper extremity strength, sensation and gait, and you confirm that [subject] is very tender at their surgical site. Per the medication administration record, [subject] has needed 1 dose of oxycodone every 4 hours, along with acetaminophen and cyclobenzaprine. ",
        "Patient D is a 55-year-old [race] [gender] presenting with 9 months of progressive low-back and leg pain. Patient D states that [possessive] pain is worse after prolonged ambulation and radiates to [possessive] right buttock and thigh. The pain has become constant in the last 2 months and [subject] is unable to function effectively at [possessive] job because of the pain. Patient D has tried physical therapy and conservative pain medications including epidural steroid injections and NSAIDs. On physical exam, you observe that [possessive] pain is significantly worse with a right straight-leg raise and there are no focal deficits. MRI of the lumbar spine reveals grade-2 spondylolisthesis at L4-L5 with degenerative changes contributing to severe canal and neuroforaminal stenosis. Flexion-extension X rays show dynamic instability of L4 on L5.",
        "Patient D is a 60-year-old [race] [gender] with history of prior L5-S1 posterior lumbar fusion who presents with 8 months of new thigh pain. Patient D states that [subject] was previously in good health and pain-free since [possessive] last surgery, but for the last 8 months, [subject] has had progressive discomfort in [possessive] anterior thighs. This discomfort has transformed into severe pain in the last 3 months, which [subject] says is worse after prolonged ambulation and often keeps [possessive] from sleeping. When you examine Patient D, you discover bilateral hip flexor weakness and diminished sensation over [possessive] anterior thighs and knees. MRI and CT of the lumbar spine show intact instrumentation at L5-S1, but with new degenerative changes at L4-L5 resulting in canal and neuroforaminal stenosis. The findings are consistent with adjacent-level disease. ",
        "Patient D is a 57-year-old [race] [gender] who is post-op day 5 status-post L4-L5 transforaminal lumbar interbody fusion for severe back and leg pain. There were no surgical complications, and [subject] continues to have low back and leg pain though [subject] endorses steady improvement in [possessive] symptoms. When you examine Patient D, you confirm that [subject] has no new neurological deficits, and does have persistent pain at [possessive] lumbar surgical site. Upon checking the medication administration record, you note that [subject] has required acetaminophen and oxycodone every 6 hours for pain control. ",
        "Patient D is a 62-year-old [race] [gender] who is post-op day 3 status-post bi-frontal craniotomy for resection of an olfactory groove meningioma. Surgery was uncomplicated, and [subject] has been healing well with no new deficits beyond expected loss of smell. Patient D is meeting all milestones for discharge, however [subject] continues to have significant head pain at the site of [possessive] craniotomy. During your pre-discharge physical examination, you note that there is expected frontal scalp swelling and tenderness to palpation of [possessive] incision site. You review the medication administration record and see that [subject] has required 1 dose of hydrocodone every 6 hours for pain control.",
        "Patient D is a 63-year-old [race] [gender] with history of renal cell carcinoma who presents with acute neck pain. Patient D says [subject] has had neck discomfort for several weeks, but It suddenly became worse today. Patient D endorses severe pain that is worse with flexion and extension of [possessive] neck, and [subject] has had no relief with ibuprofen. During [possessive] physical examination, [subject] does not have any neurological deficits, but you note tenderness to palpation of the back of [possessive] neck. ",
        "Patient D is a 75-year-old [race] [gender] with history of lung cancer and 4 months of progressive low-back and right leg pain. Patient D describes the pain as a dull ache that shoots down [possessive] right leg in no particular pattern or frequency. Recently, [subject] started experiencing intermittent numbness throughout [possessive] right leg. Overall, [subject] believes the pain is becoming more frequent and intolerable, as it now interrupts [possessive] sleep nightly and acetaminophen and ibuprofen have not helped. On your physical examination, you find that Patient D has diffusely diminished sensation in [possessive] leg, but [possessive] strength and reflexes are normal. Upon reviewing [possessive] lumbar spine MRI, you identify diffuse leptomeningeal disease. ",
        "Patient D is a 62-year-old [race] [gender] who presents to your emergency room with complaints of chest pain. Patient D has poorly controlled hypertension and several coronary artery disease risk factors. Patient D's pain is mid-sternal and intermittent, ranging from 4-8/10 and does not correlate with [possessive] activity level. Patient D denies any radiation of [possessive] pain, or any exacerbation with deep inspiration. Patient D's EKG shows mild sinus tachycardia and ST segment elevation in the inferior leads. Labs are remarkable for an elevated serum troponin.",
    ]

    df_train = df[~df.vignette.isin(test_reserve_vignette)]
    df_test = df[df.vignette.isin(test_reserve_vignette)]

    splits = DatasetDict({
        'train': Dataset.from_pandas(df_train),
        'test': Dataset.from_pandas(df_test),
    })

    print(f"train = {df_train.shape[0]}")
    print(f"test = {df_test.shape[0]}")

    # splits.save_to_disk("local/data/processed")
    if persist_path:
        splits.save_to_disk(persist_path)

    return splits
