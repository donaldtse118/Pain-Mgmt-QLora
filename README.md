# Pain-Mgmt-QLora

Download data from 
https://physionet.org/content/q-pain/1.0.0/

wget -r -N -c -np https://physionet.org/files/q-pain/1.0.0/

- Windows Git Bash,
 - get wget here : https://eternallybored.org/misc/wget/

 

From ChatGPT
 | Pain Type               | Opioid Use Recommended? | Likely Dose/Duration                           |
| ----------------------- | ----------------------- | ---------------------------------------------- |
| ✅ Acute + Cancer        | Yes                     | IV or oral opioids, **high if severe**         |
| ✅ Chronic + Cancer      | Yes                     | Oral opioids, **low–moderate, long term**      |
| 🟡 Acute + Non-cancer   | Sometimes               | **Short-term**, low dose (e.g. post-op)        |
| 🔥 Chronic + Non-cancer | **Cautious** (risk ↑)   | Try non-opioids first; opioids are last resort |
| ❌ Minor Post-op pain    | Often **No**            | NSAIDs or paracetamol sufficient               |

| Drug              | Route         | Best For                        | Typical Duration/Dose                      |
| ----------------- | ------------- | ------------------------------- | ------------------------------------------ |
| **Hydromorphone** | IV injection  | Acute, hospital-grade pain      | 0.5–1 mg per dose                          |
| **Morphine**      | IV or oral    | Cancer, trauma, palliative care | 1–4 weeks or long-term maintenance         |
| **Hydrocodone**   | Oral          | Dental, mild surgical pain      | 1 week typical, avoid long-term use        |
| **Oxycodone**     | Oral          | Cancer, chronic severe pain     | 1–4 weeks (short-term) or extended-release |
| **"Opioids"**     | General class | Context-dependent               | Follow drug-specific guidelines            |


| Drug                   |   acute\_cancer  |   acute\_non-cancer  |     chronic\_cancer     |   chronic\_non-cancer   |            post\_op           |
| ---------------------- | :--------------: | :------------------: | :---------------------: | :---------------------: | :---------------------------: |
| **Hydromorphone (IV)** |  ✅ **Best fit**  |      ⚠️ Rare use     | ✅ In-patient palliative |         ❌ Avoid         | ⚠️ Rare use (if extreme pain) |
| **Morphine**           |    ✅ Standard    |    ⚠️ Cautious use   |        ✅ Standard       | ⚠️ Cautious (try avoid) |       ✅ Common if severe      |
| **Hydrocodone**        |    ❌ Too weak    | ✅ Mild/moderate pain |     ⚠️ Not preferred    |    ⚠️ Short-term only   |            ✅ Common           |
| **Oxycodone**          | ✅ Outpatient use |   ⚠️ Risky if long   |       ✅ Often used      |     ⚠️ Controversial    |        ✅ If NSAIDs fail       |
| **Opioids (general)**  |         ✅        |    ⚠️ Case-by-case   |            ✅            |     ⚠️ Not 1st line     |        ✅ For short-term       |


## Drug usage on different pain type on collected data
| Drug                  | Pain Type          | `Yes` Count | `No` Count                          |
| --------------------- | ------------------ | ----------- | ----------------------------------- |
| **Hydrocodone**       | chronic cancer     | 0           | **1**        ⚠️ uncommon combo      |
|                       | chronic non cancer | **10**      | 0           ⚠️ overconfident        |
|                       | post op            | **6**       | **1**        ✅                      |
| **IV Hydromorphone**  | acute cancer       | **10**      | **1**        ✅                      |
|                       | acute non cancer   | **10**      | **1**        ⚠️ too generous        |
| **Morphine**          | chronic cancer     | **10**      | 0           ⚠️ needs more variation |
| **Opioids (general)** | chronic non cancer | 0           | **1**        ⚠️ single datapoint    |
| **Oxycodone**         | post op            | **4**       | 0           ⚠️ all-positive bias    |


## Evaluation Strategy 
to extract those information

| Factor           | True Value       | Predicted Value | Match       |
| ---------------- | ---------------- | --------------- | ----------- |
| Pain severity    | severe           | severe          | ✅           |
| Clinical context | fracture, cancer | fracture        | ✅ (partial) |
| Med failure      | NSAIDs failed    | not mentioned   | ❌           |
