---
layout: page
title: Model Performance
keywords: stanza, system performance
permalink: '/performance.html'
nav_order: 2
parent: Models
datatable: true
---

Here we report the performance of Stanza's pretrained models on all supported languages. Again, performances of models for tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging and dependency parsing are reported on the Universal Dependencies (UD) treebanks, while performances of the NER models are reported separately.

## System Performance on UD Treebanks

In the table below you can find the performance of Stanza's pretrained UD models as of version 1.3 on UD 2.8.
Performance as of v1.0 on UD 2.5 [is also available](v100performance.md).  All models are trained and tested with the Universal Dependencies v2.8 treebanks.
Note that all scores reported are from an end-to-end evaluation on the official test sets (from raw text to the full CoNLL-U file), and are generated with the CoNLL 2018 UD shared task official evaluation script. For detailed interpretation of these scores and the evaluation scripts we used, please refer to the [CoNLL 2018 UD Shared Task Evaluation](https://universaldependencies.org/conll18/evaluation.html) page. For details on how we handled treebanks with no training data, please refer to [our CoNLL 2018 system description paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf).

For more detailed results and better viewing experience, please access the results in [this Google Spreadsheet](https://docs.google.com/spreadsheets/d/1t9h8QxjYA2XK4qs9Q7R4wiOwQHPykpBTjH8PGj_on5Y/edit?usp=sharing).

| Treebank | Tokens | Sentences | Words | UPOS | XPOS | UFeats | AllTags | Lemmas | UAS | LAS | CLAS | MLAS | BLEX |
| :------- | :----- | :-------- | :---- | :--- | :--- | :----- | :------ | :----- | :-- | :-- | :--- | :--- | :--- |
|Macro Avg  |  99.52  |  88.11  |  99.13  |  94.81  |  94.40  |  92.71  |  90.01  |  95.05  |  83.62  |  79.04  |  75.08  |  68.22  |  71.83|
|UD_Afrikaans-AfriBooms  |  99.93  |  100.00  |  99.93  |  97.97  |  93.97  |  97.22  |  93.88  |  97.39  |  87.38  |  83.57  |  76.66  |  72.07  |  73.00|
|UD_Ancient_Greek-PROIEL  |  100.00  |  52.02  |  100.00  |  97.38  |  97.74  |  92.20  |  91.03  |  97.21  |  81.27  |  77.38  |  72.60  |  62.22  |  70.27|
|UD_Ancient_Greek-Perseus  |  99.99  |  98.93  |  99.99  |  92.36  |  84.95  |  91.03  |  84.75  |  87.58  |  79.26  |  73.65  |  67.83  |  53.86  |  56.75|
|UD_Arabic-PADT  |  99.98  |  82.87  |  98.61  |  95.61  |  92.49  |  92.64  |  92.32  |  94.16  |  84.40  |  79.94  |  76.87  |  71.15  |  73.54|
|UD_Armenian-ArmTDP  |  99.92  |  99.46  |  99.48  |  95.70  |  99.48  |  91.66  |  90.63  |  94.91  |  83.82  |  77.95  |  73.50  |  65.88  |  69.97|
|UD_Basque-BDT  |  99.99  |  100.00  |  99.99  |  96.09  |  99.99  |  92.93  |  91.19  |  96.26  |  86.19  |  82.76  |  81.12  |  73.28  |  77.95|
|UD_Belarusian-HSE  |  98.48  |  88.12  |  98.48  |  96.75  |  94.90  |  91.97  |  89.98  |  94.07  |  85.29  |  82.80  |  79.43  |  72.79  |  74.80|
|UD_Bulgarian-BTB  |  99.93  |  96.29  |  99.93  |  98.83  |  96.34  |  97.57  |  95.88  |  97.31  |  93.30  |  90.16  |  86.76  |  83.42  |  83.58|
|UD_Catalan-AnCora  |  99.97  |  99.59  |  99.95  |  98.77  |  99.95  |  98.38  |  97.95  |  98.57  |  93.17  |  90.97  |  86.20  |  83.84  |  85.38|
|UD_Chinese-GSD  |  93.99  |  98.80  |  93.99  |  90.26  |  90.05  |  93.18  |  89.23  |  93.99  |  74.29  |  70.98  |  68.31  |  64.41  |  68.31|
|UD_Chinese-GSDSimp  |  93.89  |  98.80  |  93.89  |  90.04  |  89.80  |  93.06  |  88.98  |  93.89  |  75.14  |  71.91  |  69.31  |  65.33  |  69.31|
|UD_Classical_Chinese-Kyoto  |  97.17  |  41.73  |  97.17  |  87.80  |  86.69  |  90.62  |  84.43  |  96.93  |  68.21  |  62.73  |  61.36  |  59.03  |  61.26|
|UD_Coptic-Scriptorium  |  100.00  |  34.83  |  87.13  |  83.96  |  83.84  |  84.86  |  82.97  |  84.01  |  64.45  |  62.44  |  53.78  |  50.65  |  52.46|
|UD_Croatian-SET  |  99.94  |  97.61  |  99.94  |  97.82  |  94.68  |  95.08  |  93.96  |  96.59  |  90.44  |  86.79  |  83.92  |  77.43  |  80.09|
|UD_Czech-CAC  |  99.92  |  100.00  |  99.81  |  98.53  |  94.34  |  93.08  |  92.19  |  97.65  |  91.11  |  88.64  |  86.42  |  79.39  |  84.40|
|UD_Czech-CLTT  |  99.19  |  97.79  |  99.12  |  98.13  |  91.43  |  91.60  |  90.80  |  96.70  |  86.45  |  83.93  |  81.05  |  72.62  |  78.98|
|UD_Czech-FicTree  |  99.98  |  98.95  |  99.97  |  98.31  |  95.12  |  96.01  |  94.60  |  98.27  |  93.26  |  90.73  |  88.25  |  82.88  |  86.15|
|UD_Czech-PDT  |  99.98  |  94.99  |  99.98  |  98.60  |  95.39  |  94.63  |  93.68  |  98.43  |  92.25  |  90.18  |  88.46  |  82.10  |  86.82|
|UD_Danish-DDT  |  99.90  |  91.65  |  99.90  |  97.64  |  99.90  |  97.35  |  96.36  |  96.97  |  86.63  |  84.36  |  81.33  |  77.30  |  78.09|
|UD_Dutch-Alpino  |  99.52  |  88.07  |  99.52  |  95.87  |  94.48  |  96.00  |  93.74  |  94.03  |  89.43  |  86.41  |  81.73  |  76.35  |  74.28|
|UD_Dutch-LassySmall  |  99.88  |  80.77  |  99.88  |  95.98  |  95.11  |  96.33  |  94.30  |  95.24  |  87.06  |  83.34  |  77.26  |  73.20  |  72.12|
|UD_English-EWT  |  99.36  |  89.09  |  99.11  |  95.64  |  95.40  |  96.23  |  94.02  |  96.91  |  87.41  |  84.91  |  81.60  |  77.18  |  79.35|
|UD_English-GUM  |  99.25  |  88.98  |  99.42  |  96.15  |  96.15  |  95.39  |  94.00  |  97.55  |  88.30  |  85.68  |  81.57  |  76.60  |  79.49|
|UD_English-LinES  |  99.96  |  90.05  |  99.96  |  96.92  |  96.17  |  96.71  |  94.07  |  98.21  |  86.80  |  82.92  |  78.51  |  73.81  |  76.71|
|UD_English-ParTUT  |  99.66  |  100.00  |  99.57  |  96.11  |  95.79  |  95.26  |  93.97  |  97.20  |  89.09  |  86.36  |  81.09  |  75.34  |  78.61|
|UD_Estonian-EDT  |  99.95  |  93.10  |  99.95  |  97.21  |  98.07  |  95.86  |  94.51  |  95.91  |  87.15  |  84.43  |  82.93  |  77.78  |  78.75|
|UD_Estonian-EWT  |  98.78  |  78.96  |  98.78  |  92.55  |  94.45  |  91.43  |  88.50  |  89.81  |  76.97  |  71.94  |  69.01  |  62.15  |  61.63|
|UD_Faroese-FarPaHC  |  99.72  |  94.45  |  99.70  |  97.00  |  91.76  |  93.15  |  91.20  |  99.70  |  84.04  |  79.68  |  71.74  |  64.57  |  71.74|
|UD_Finnish-FTB  |  100.00  |  90.38  |  99.98  |  95.47  |  95.15  |  96.52  |  93.94  |  95.99  |  89.55  |  86.74  |  84.03  |  79.55  |  80.97|
|UD_Finnish-TDT  |  99.80  |  91.29  |  99.76  |  96.94  |  97.76  |  95.41  |  94.45  |  94.70  |  88.89  |  86.54  |  85.09  |  79.97  |  80.28|
|UD_French-GSD  |  99.73  |  95.77  |  99.46  |  97.42  |  99.45  |  97.55  |  96.78  |  97.88  |  91.98  |  89.60  |  85.18  |  81.98  |  83.45|
|UD_French-ParTUT  |  99.88  |  100.00  |  99.42  |  96.54  |  96.15  |  94.00  |  93.31  |  96.04  |  91.42  |  88.96  |  84.18  |  75.65  |  79.95|
|UD_French-Sequoia  |  99.86  |  90.99  |  99.55  |  98.39  |  99.55  |  97.71  |  97.25  |  98.22  |  91.14  |  88.90  |  85.35  |  82.68  |  83.92|
|UD_French-Spoken  |  99.97  |  23.44  |  99.38  |  95.63  |  96.83  |  92.82  |  88.47  |  95.78  |  77.23  |  70.95  |  62.21  |  56.11  |  60.61|
|UD_Galician-CTG  |  99.86  |  98.78  |  99.25  |  97.14  |  96.91  |  99.09  |  96.63  |  97.85  |  85.06  |  82.53  |  77.12  |  70.87  |  75.77|
|UD_Galician-TreeGal  |  99.61  |  88.81  |  98.51  |  94.13  |  91.86  |  93.20  |  90.97  |  94.64  |  78.06  |  72.99  |  65.72  |  58.99  |  62.03|
|UD_German-GSD  |  99.49  |  82.81  |  99.47  |  94.40  |  96.93  |  89.28  |  84.68  |  96.22  |  85.29  |  80.66  |  75.50  |  59.20  |  71.21|
|UD_German-HDT  |  100.00  |  97.10  |  100.00  |  97.95  |  97.83  |  91.32  |  90.87  |  97.50  |  94.71  |  92.26  |  88.20  |  75.97  |  85.16|
|UD_Gothic-PROIEL  |  100.00  |  40.62  |  100.00  |  96.10  |  96.69  |  90.69  |  88.93  |  96.02  |  75.21  |  69.45  |  66.82  |  57.95  |  64.51|
|UD_Greek-GDT  |  99.94  |  93.08  |  99.93  |  97.64  |  97.64  |  94.93  |  94.27  |  96.28  |  91.53  |  89.14  |  84.40  |  78.07  |  79.63|
|UD_Hebrew-HTB  |  100.00  |  100.00  |  92.82  |  90.15  |  90.15  |  88.78  |  88.05  |  89.93  |  79.62  |  76.93  |  70.29  |  63.84  |  67.25|
|UD_Hindi-HDTB  |  100.00  |  99.29  |  100.00  |  97.55  |  97.07  |  93.87  |  91.98  |  96.80  |  94.74  |  91.67  |  88.12  |  78.29  |  86.82|
|UD_Hungarian-Szeged  |  99.92  |  97.45  |  99.92  |  96.00  |  99.92  |  93.62  |  92.73  |  94.19  |  84.19  |  79.23  |  77.23  |  68.82  |  71.61|
|UD_Icelandic-IcePaHC  |  99.91  |  93.18  |  99.87  |  96.23  |  92.02  |  89.72  |  84.59  |  95.62  |  87.09  |  83.14  |  77.64  |  64.18  |  73.82|
|UD_Icelandic-Modern  |  99.97  |  98.83  |  99.97  |  98.59  |  97.53  |  98.01  |  97.32  |  98.70  |  93.42  |  91.91  |  89.54  |  87.27  |  88.29|
|UD_Indonesian-CSUI  |  99.58  |  92.90  |  99.08  |  95.69  |  96.09  |  96.65  |  95.39  |  96.12  |  81.40  |  76.39  |  73.19  |  70.42  |  70.58|
|UD_Indonesian-GSD  |  99.98  |  94.13  |  99.98  |  93.85  |  94.54  |  95.75  |  89.12  |  99.65  |  86.53  |  80.36  |  77.40  |  69.03  |  77.06|
|UD_Irish-IDT  |  99.88  |  96.91  |  99.88  |  96.08  |  94.87  |  89.34  |  86.55  |  95.23  |  87.20  |  81.66  |  76.40  |  63.24  |  71.74|
|UD_Italian-ISDT  |  99.94  |  99.07  |  99.83  |  97.93  |  97.88  |  97.79  |  97.15  |  98.01  |  92.78  |  90.99  |  86.71  |  83.80  |  84.33|
|UD_Italian-ParTUT  |  99.81  |  100.00  |  99.79  |  97.79  |  97.62  |  97.46  |  96.63  |  97.60  |  91.99  |  89.57  |  83.89  |  80.55  |  81.27|
|UD_Italian-PoSTWITA  |  99.63  |  66.72  |  99.36  |  96.16  |  95.97  |  96.25  |  94.90  |  96.49  |  83.24  |  78.85  |  73.11  |  69.61  |  71.00|
|UD_Italian-TWITTIRO  |  99.23  |  55.24  |  98.91  |  94.46  |  93.86  |  93.25  |  91.16  |  92.98  |  79.14  |  72.55  |  63.10  |  57.53  |  57.34|
|UD_Italian-VIT  |  99.98  |  95.99  |  99.53  |  97.41  |  96.49  |  97.27  |  95.55  |  97.99  |  89.25  |  86.02  |  80.38  |  76.61  |  78.57|
|UD_Japanese-GSD  |  96.79  |  99.72  |  96.79  |  95.58  |  95.01  |  96.76  |  94.77  |  96.28  |  88.50  |  87.64  |  82.80  |  81.40  |  82.42|
|UD_Korean-GSD  |  99.85  |  95.74  |  99.85  |  96.09  |  90.79  |  99.59  |  88.52  |  93.65  |  87.46  |  83.71  |  81.61  |  79.27  |  76.22|
|UD_Korean-Kaist  |  100.00  |  99.93  |  100.00  |  95.58  |  86.27  |  100.00  |  86.27  |  94.25  |  88.51  |  86.53  |  84.15  |  80.82  |  79.07|
|UD_Latin-ITTB  |  99.99  |  80.73  |  99.99  |  97.55  |  95.78  |  95.73  |  93.57  |  99.07  |  88.76  |  86.66  |  85.47  |  80.49  |  85.06|
|UD_Latin-LLCT  |  100.00  |  99.49  |  100.00  |  99.50  |  96.73  |  96.79  |  96.41  |  98.08  |  95.97  |  94.81  |  93.71  |  89.73  |  91.72|
|UD_Latin-PROIEL  |  100.00  |  42.28  |  100.00  |  96.98  |  97.13  |  91.26  |  90.46  |  96.71  |  76.99  |  72.86  |  69.73  |  60.95  |  67.82|
|UD_Latin-Perseus  |  100.00  |  98.99  |  100.00  |  90.97  |  78.18  |  82.44  |  77.53  |  82.70  |  71.24  |  61.58  |  57.06  |  44.55  |  45.88|
|UD_Latin-UDante  |  99.96  |  98.45  |  99.64  |  89.58  |  75.06  |  79.53  |  72.92  |  87.34  |  68.30  |  58.77  |  48.61  |  35.37  |  41.99|
|UD_Latvian-LVTB  |  99.83  |  98.97  |  99.83  |  96.42  |  88.58  |  93.69  |  87.99  |  95.61  |  89.09  |  85.89  |  83.63  |  75.40  |  79.82|
|UD_Lithuanian-ALKSNIS  |  99.94  |  89.38  |  99.94  |  93.54  |  85.88  |  87.15  |  84.55  |  92.78  |  78.81  |  73.66  |  71.02  |  59.69  |  65.84|
|UD_Lithuanian-HSE  |  98.87  |  51.11  |  98.87  |  82.89  |  81.85  |  74.39  |  69.00  |  77.03  |  50.66  |  39.41  |  34.57  |  24.40  |  27.31|
|UD_Maltese-MUDT  |  99.83  |  85.38  |  99.83  |  95.79  |  95.70  |  99.83  |  95.39  |  99.83  |  83.23  |  77.97  |  70.41  |  66.61  |  70.41|
|UD_Marathi-UFAL  |  97.59  |  92.63  |  94.85  |  76.23  |  94.85  |  64.22  |  59.80  |  76.72  |  69.12  |  57.84  |  50.52  |  29.65  |  41.34|
|UD_Naija-NSC  |  99.97  |  100.00  |  99.97  |  97.59  |  99.97  |  98.93  |  97.19  |  99.02  |  92.43  |  89.49  |  88.52  |  86.50  |  88.09|
|UD_North_Sami-Giella  |  99.71  |  99.83  |  99.71  |  90.44  |  92.46  |  87.10  |  82.80  |  88.12  |  72.87  |  66.66  |  64.14  |  57.51  |  56.48|
|UD_Norwegian-Bokmaal  |  99.87  |  97.65  |  99.87  |  98.04  |  99.87  |  97.07  |  96.29  |  98.10  |  92.16  |  90.35  |  87.86  |  83.80  |  85.69|
|UD_Norwegian-Nynorsk  |  99.96  |  94.29  |  99.96  |  97.70  |  99.96  |  96.77  |  95.82  |  97.73  |  91.49  |  89.59  |  87.09  |  82.33  |  84.41|
|UD_Norwegian-NynorskLIA  |  100.00  |  99.69  |  100.00  |  96.09  |  100.00  |  95.26  |  93.11  |  97.55  |  78.05  |  73.25  |  68.10  |  62.14  |  65.76|
|UD_Old_Church_Slavonic-PROIEL  |  100.00  |  50.67  |  100.00  |  96.51  |  96.83  |  90.59  |  89.60  |  95.62  |  79.61  |  75.10  |  74.54  |  65.95  |  71.79|
|UD_Old_East_Slavic-RNC  |  97.87  |  81.60  |  97.87  |  86.85  |  87.11  |  66.30  |  59.36  |  76.20  |  60.86  |  51.86  |  40.82  |  25.80  |  30.41|
|UD_Old_East_Slavic-TOROT  |  100.00  |  32.78  |  100.00  |  93.40  |  93.55  |  86.43  |  84.28  |  90.83  |  72.47  |  66.40  |  63.07  |  52.46  |  58.51|
|UD_Old_French-SRCMF  |  100.00  |  100.00  |  100.00  |  96.06  |  95.89  |  97.61  |  95.31  |  100.00  |  90.68  |  85.71  |  81.86  |  79.00  |  81.86|
|UD_Persian-PerDT  |  99.96  |  99.86  |  99.63  |  97.03  |  96.96  |  97.29  |  95.11  |  98.64  |  92.67  |  90.39  |  88.29  |  84.57  |  87.21|
|UD_Persian-Seraji  |  100.00  |  99.25  |  99.67  |  97.19  |  97.26  |  97.36  |  96.80  |  97.81  |  89.80  |  86.51  |  83.51  |  81.48  |  81.82|
|UD_Polish-LFG  |  99.91  |  99.91  |  99.91  |  98.43  |  94.35  |  95.42  |  93.67  |  96.55  |  95.75  |  93.78  |  92.00  |  86.69  |  87.98|
|UD_Polish-PDB  |  99.87  |  98.26  |  99.84  |  98.26  |  94.10  |  94.38  |  93.33  |  97.28  |  93.28  |  90.97  |  88.86  |  82.20  |  85.85|
|UD_Portuguese-Bosque  |  99.81  |  91.59  |  99.74  |  97.07  |  99.74  |  96.14  |  94.73  |  97.79  |  91.09  |  87.99  |  83.05  |  76.43  |  80.71|
|UD_Portuguese-GSD  |  99.92  |  97.84  |  99.82  |  98.12  |  98.10  |  99.66  |  98.03  |  99.14  |  91.59  |  87.85  |  83.78  |  80.00  |  82.96|
|UD_Romanian-Nonstandard  |  98.86  |  97.53  |  98.86  |  95.28  |  90.57  |  89.42  |  88.03  |  94.81  |  88.71  |  84.47  |  80.12  |  66.45  |  76.35|
|UD_Romanian-RRT  |  99.73  |  95.40  |  99.73  |  97.63  |  96.95  |  97.16  |  96.87  |  97.52  |  90.56  |  86.51  |  82.60  |  79.19  |  80.45|
|UD_Romanian-SiMoNERo  |  99.62  |  100.00  |  99.62  |  97.86  |  97.41  |  97.05  |  96.85  |  98.57  |  92.97  |  91.10  |  88.27  |  84.28  |  87.14|
|UD_Russian-GSD  |  99.65  |  97.16  |  99.65  |  97.38  |  97.18  |  93.08  |  92.19  |  95.03  |  89.01  |  84.81  |  82.52  |  75.21  |  77.41|
|UD_Russian-SynTagRus  |  99.59  |  98.95  |  99.59  |  98.27  |  99.59  |  95.99  |  95.67  |  97.46  |  92.79  |  91.13  |  89.51  |  85.22  |  87.15|
|UD_Russian-Taiga  |  99.00  |  88.30  |  99.00  |  95.78  |  99.00  |  92.73  |  91.53  |  94.16  |  80.90  |  76.48  |  73.26  |  66.84  |  68.74|
|UD_Sanskrit-Vedic  |  100.00  |  33.21  |  100.00  |  88.88  |  100.00  |  80.46  |  77.01  |  88.52  |  61.59  |  49.33  |  48.65  |  42.45  |  45.49|
|UD_Scottish_Gaelic-ARCOSG  |  99.72  |  65.37  |  98.85  |  94.27  |  89.12  |  90.86  |  88.11  |  95.33  |  82.84  |  77.61  |  71.47  |  64.52  |  68.34|
|UD_Serbian-SET  |  99.99  |  99.33  |  99.99  |  98.32  |  94.08  |  94.41  |  93.64  |  96.32  |  91.84  |  88.92  |  86.50  |  79.26  |  82.36|
|UD_Slovenian-SSJ  |  99.90  |  98.10  |  99.90  |  98.33  |  95.13  |  95.43  |  94.62  |  97.07  |  92.72  |  90.97  |  88.37  |  83.21  |  84.98|
|UD_Slovenian-SST  |  100.00  |  27.46  |  100.00  |  93.26  |  87.67  |  87.64  |  84.63  |  94.15  |  62.60  |  55.26  |  49.51  |  42.52  |  46.72|
|UD_Spanish-AnCora  |  99.97  |  98.32  |  99.95  |  98.75  |  99.95  |  98.35  |  97.89  |  99.20  |  92.13  |  89.94  |  85.62  |  83.19  |  84.86|
|UD_Spanish-GSD  |  99.91  |  95.60  |  99.82  |  96.61  |  99.82  |  96.43  |  94.44  |  98.36  |  89.78  |  87.07  |  81.93  |  74.10  |  79.79|
|UD_Swedish-LinES  |  99.92  |  87.20  |  99.92  |  96.65  |  94.41  |  90.09  |  87.26  |  96.64  |  88.06  |  83.86  |  80.48  |  67.87  |  77.11|
|UD_Swedish-Talbanken  |  99.98  |  98.03  |  99.98  |  97.66  |  96.46  |  96.68  |  95.59  |  97.51  |  88.64  |  85.51  |  83.08  |  78.53  |  80.31|
|UD_Swedish_Sign_Language-SSLC  |  100.00  |  3.39  |  100.00  |  77.66  |  78.01  |  100.00  |  74.82  |  100.00  |  23.05  |  13.12  |  13.65  |  10.84  |  13.65|
|UD_Tamil-TTB  |  99.32  |  98.76  |  92.12  |  79.92  |  79.31  |  81.59  |  74.60  |  84.48  |  64.47  |  56.06  |  53.52  |  45.73  |  48.60|
|UD_Telugu-MTG  |  99.79  |  98.97  |  99.79  |  93.14  |  93.14  |  98.96  |  93.14  |  99.79  |  90.64  |  81.22  |  76.12  |  71.93  |  76.12|
|UD_Turkish-BOUN  |  99.38  |  90.25  |  98.62  |  90.70  |  91.64  |  92.18  |  87.10  |  93.56  |  75.95  |  69.64  |  67.52  |  58.30  |  63.37|
|UD_Turkish-FrameNet  |  100.00  |  100.00  |  100.00  |  94.89  |  100.00  |  92.30  |  91.48  |  93.05  |  90.93  |  80.85  |  78.23  |  69.81  |  71.27|
|UD_Turkish-IMST  |  99.94  |  97.87  |  98.23  |  92.49  |  93.34  |  90.86  |  88.91  |  95.17  |  71.74  |  65.46  |  61.84  |  55.90  |  60.04|
|UD_Turkish-Kenet  |  99.99  |  98.87  |  99.99  |  93.44  |  99.99  |  91.43  |  90.52  |  92.71  |  82.72  |  69.68  |  68.90  |  60.02  |  63.10|
|UD_Turkish-Penn  |  100.00  |  89.84  |  100.00  |  94.93  |  100.00  |  92.60  |  91.93  |  93.58  |  86.34  |  76.70  |  74.72  |  64.12  |  68.53|
|UD_Turkish-Tourism  |  100.00  |  99.93  |  100.00  |  99.01  |  100.00  |  94.97  |  94.78  |  98.95  |  95.90  |  90.28  |  87.64  |  80.49  |  86.63|
|UD_Turkish_German-SAGT  |  99.96  |  99.07  |  99.04  |  85.71  |  99.04  |  75.48  |  69.88  |  90.64  |  64.79  |  52.30  |  46.70  |  33.94  |  42.83|
|UD_Ukrainian-IU  |  99.78  |  97.48  |  99.75  |  96.74  |  92.40  |  92.47  |  91.29  |  96.23  |  87.83  |  84.71  |  81.40  |  73.84  |  77.77|
|UD_Urdu-UDTB  |  100.00  |  98.88  |  100.00  |  94.61  |  92.74  |  84.43  |  80.59  |  95.64  |  88.13  |  82.53  |  76.88  |  59.18  |  74.59|
|UD_Uyghur-UDT  |  99.46  |  84.72  |  99.46  |  88.76  |  91.17  |  87.52  |  79.87  |  95.16  |  74.74  |  62.91  |  56.66  |  45.40  |  53.67|
|UD_Vietnamese-VTB  |  89.22  |  95.45  |  89.22  |  81.65  |  79.91  |  89.03  |  79.88  |  89.22  |  56.01  |  50.41  |  47.50  |  44.48  |  47.50|
|UD_Welsh-CCG  |  99.84  |  97.89  |  99.29  |  93.97  |  92.58  |  89.06  |  86.25  |  92.56  |  83.21  |  76.16  |  69.77  |  58.45  |  62.76|
|UD_Western_Armenian-ArmTDP  |  99.81  |  98.11  |  99.69  |  92.38  |  99.69  |  85.21  |  84.12  |  93.33  |  78.83  |  70.11  |  60.89  |  47.62  |  56.10|
|UD_Wolof-WTB  |  99.93  |  92.31  |  99.37  |  93.86  |  93.61  |  92.83  |  90.76  |  93.96  |  83.52  |  77.41  |  71.47  |  64.41  |  67.16|
{: .compact #conll18-results .datatable }

## System Performance on NER Corpora

In the table below you can find the performance of Stanza's pretrained NER models. All numbers reported are micro-averaged F1 scores. We used canonical train/dev/test splits for all datasets except for the WikiNER datasets, for which we used random splits.  The Ukrainian model and its score [was provided by a user](https://github.com/stanfordnlp/stanza/issues/319).

| Language | Corpus | # Types | F1 |
| :------- | :----- | :-------- | :---- |
| Afrikaans | NCHLT | 4 | 80.08 |
| Arabic | AQMAR | 4 | 74.3 |
| Bulgarian *New in 1.2.1* | BSNLP 2019 | 5 | 83.21 |
| Chinese | OntoNotes | 18 | 79.2 |
| Dutch | CoNLL02 | 4 | 89.2 |
| Dutch | WikiNER | 4 | 94.8 |
| English | CoNLL03 | 4 | 92.1 |
| English | OntoNotes | 18 | 88.8 |
| Finnish *New in 1.2.1* | Turku | 6 | 87.04 |
| French | WikiNER | 4 | 92.9 |
| German | CoNLL03 | 4 | 81.9 |
| German | GermEval2014 | 4 | 85.2 |
| Hungarian *New in 1.2.1* | Combined | 4 | - |
| Italian *New in 1.2.3* | FBK | 3 | 87.92 |
| Myanmar *New in 1.3.1* | UCSY | 7 | 95.86 |
| Russian | WikiNER | 4 | 92.9 |
| Spanish | CoNLL02 | 4 | 88.1 |
| Spanish | AnCora | 4 | 88.6 |
| Ukrainian | languk | 4 | 86.05 |
| Vietnamese *New in 1.2.1* | VLSP | 4 | 82.44 |

### Notes on NER Corpora

We have provided links to all NER datasets used to train the released models on our [available NER models page](available_models.md#available-ner-models). Here we provide notes on how to find several of these corpora:

- **Afrikaans**: The Afrikaans data is part of [the NCHLT corpus of South African languages](https://repo.sadilar.org/handle/20.500.12185/299).  Van Huyssteen, G.B., Puttkammer, M.J., Trollip, E.B., Liversage, J.C., Eiselen, R. 2016. [NCHLT Afrikaans Named Entity Annotated Corpus. 1.0](https://hdl.handle.net/20.500.12185/299).


- **Bulgarian**: The Bulgarian BSNLP 2019 data is available from [the shared task page](http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html). You can also find their [dataset description paper](https://www.aclweb.org/anthology/W19-3709/).

- **Finnish**: The Turku dataset used for Finnish NER training can be found on [the Turku NLP website](https://turkunlp.org/fin-ner.html), and they also provide [a Turku NER dataset description paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.567.pdf).

- **Hungarian**: The dataset used for training our Hungarian NER system is a combination of 3 separate datasets. Two of these datasets can be found from [this Szeged page](https://rgai.inf.u-szeged.hu/node/130), and the third can be found in [this NYTK-NerKor github repo](https://github.com/nytud/NYTK-NerKor). A dataset description paper can also be found [here](http://www.inf.u-szeged.hu/projectdirs/hlt/papers/lrec_ne-corpus.pdf).

- **Italian**: The Italian FBK dataset was licensed to us from [FBK](https://dh.fbk.eu/).  Paccosi T. and Palmero Aprosio A.  KIND: an Italian Multi-Domain Dataset for Named Entity Recognition.  LREC 2022

- **Myanmar**: The Myanmar dataset is by special request from [UCSY](https://arxiv.org/ftp/arxiv/papers/1903/1903.04739.pdf).

- **Vietnamese**: The Vietnamese VLSP dataset is available by [request from VLSP](https://vlsp.org.vn/vlsp2018/eval/ner).
