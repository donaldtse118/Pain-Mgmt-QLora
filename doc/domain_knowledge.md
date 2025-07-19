# Domain Knowledge on Opioid Drugs and Pain Management

This section summarizes key domain insights on commonly used opioids for different pain types, based on literature and our collected dataset. It helps explain the rationale behind our data preparation and model training.

---

# Common Opioid Drugs Overview
| Drug                  | Route        | Typical Use Cases                    | Typical Duration / Dose                          |
| --------------------- | ------------ | ------------------------------------ | ------------------------------------------------ |
| **Hydromorphone**     | IV injection | Acute, severe hospital-grade pain    | 0.5–1 mg per dose                                |
| **Morphine**          | IV or oral   | Cancer pain, trauma, palliative care | 1–4 weeks or long-term maintenance               |
| **Hydrocodone**       | Oral         | Dental pain, mild surgical pain      | Usually \~1 week; avoid long-term use            |
| **Oxycodone**         | Oral         | Cancer pain, chronic severe pain     | 1–4 weeks (short-term) or extended-release forms |
| **Opioids (general)** | Various      | Context-dependent                    | Follow drug-specific guidelines                  |

---

# Drugs vs. Pain Type Matrix
I considered creating a dataset for new drug-pain combinations but abandoned it due to insufficient domain expertise.

| Drug                   |   Acute Cancer   |    Acute Non-Cancer   |      Chronic Cancer     |  Chronic Non-Cancer  |       Post-Operative Pain       |
| ---------------------- | :--------------: | :-------------------: | :---------------------: | :------------------: | :-----------------------------: |
| **Hydromorphone (IV)** |    ✅ Best fit    |      ⚠️ Rare use      | ✅ In-patient palliative |        ❌ Avoid       | ⚠️ Rare use (only extreme pain) |
| **Morphine**           |    ✅ Standard    |   ⚠️ Use cautiously   |        ✅ Standard       | ⚠️ Avoid if possible |        ✅ Common if severe       |
| **Hydrocodone**        |    ❌ Too weak    |    ✅ Mild/moderate    |     ⚠️ Not preferred    |  ⚠️ Short-term only  |             ✅ Common            |
| **Oxycodone**          | ✅ Outpatient use | ⚠️ Risky if prolonged |       ✅ Often used      |   ⚠️ Controversial   |         ✅ If NSAIDs fail        |
| **Opioids (general)**  |   ✅ Applicable   |    ⚠️ Case-by-case    |       ✅ Applicable      |   ⚠️ Not first line  |         ✅ Short-term use        |

---

# Drug Usage in Collected Dataset by Pain Type
| Drug                  | Pain Type          | `Yes` Count | `No` Count |
| --------------------- | ------------------ | ----------- | ---------- |
| **Hydrocodone**       | Chronic cancer     | 0           | **1**      |
|                       | Chronic non-cancer | **10**      | 0          |
|                       | Post-operative     | **6**       | **1**      |
| **IV Hydromorphone**  | Acute cancer       | **10**      | **1**      |
|                       | Acute non-cancer   | **10**      | **1**      |
| **Morphine**          | Chronic cancer     | **10**      | 0          |
| **Opioids (general)** | Chronic non-cancer | 0           | **1**      |
| **Oxycodone**         | Post-operative     | **4**       | 0          |


