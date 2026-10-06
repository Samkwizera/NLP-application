# Error analysis - t5_best_maxlen512_weighted

## Per-class results (test)
|               |   n_train |   n_test |   baseline_f1 |   precision |   recall |   f1 |
|:--------------|----------:|---------:|--------------:|------------:|---------:|-----:|
| politics      |      1503 |      315 |          0.61 |        0.64 |     0.63 | 0.64 |
| sport         |       828 |      744 |          0.94 |        0.94 |     0.96 | 0.95 |
| economy       |       943 |      174 |          0.54 |        0.51 |     0.76 | 0.61 |
| health        |       920 |      272 |          0.58 |        0.65 |     0.67 | 0.66 |
| entertainment |       846 |      774 |          0.8  |        0.83 |     0.84 | 0.83 |
| history       |       185 |       11 |          0.19 |        0.11 |     0.82 | 0.2  |
| technology    |       127 |      103 |          0.6  |        0.63 |     0.77 | 0.69 |
| tourism       |        28 |       76 |          0.1  |        0.73 |     0.62 | 0.67 |
| culture       |        65 |      138 |          0.12 |        0.76 |     0.38 | 0.51 |
| fashion       |        31 |       91 |          0    |        0.55 |     0.36 | 0.44 |
| religion      |       380 |      256 |          0.63 |        0.78 |     0.74 | 0.76 |
| environment   |        32 |      100 |          0    |        0.88 |     0.21 | 0.34 |
| education     |        99 |       70 |          0.03 |        0.12 |     0.04 | 0.06 |
| relationship  |       357 |      340 |          0.82 |        0.78 |     0.87 | 0.82 |

## Accuracy by news source
Sources marked seen_source=True also appear in the training data.
|                           |   accuracy |    n |
|:--------------------------|-----------:|-----:|
| ('igihe.com', False)      |      0.66  | 1205 |
| ('kigalipost.com', False) |      0.607 |  671 |
| ('muhabura.rw', True)     |      0.882 | 1588 |

## Accuracy by truncation (article longer than 512 tokens)
| truncated   |   accuracy |    n |
|:------------|-----------:|-----:|
| False       |      0.787 | 1613 |
| True        |      0.72  | 1851 |

## Accuracy by confidence
| confidence_bin   |   accuracy |    n |
|:-----------------|-----------:|-----:|
| (0.0, 0.5]       |      0.257 |  148 |
| (0.5, 0.7]       |      0.438 |  329 |
| (0.7, 0.9]       |      0.512 |  521 |
| (0.9, 1.0]       |      0.873 | 2466 |

## Most common confusions
|                               |   count |
|:------------------------------|--------:|
| ('fashion', 'entertainment')  |      45 |
| ('politics', 'economy')       |      33 |
| ('education', 'relationship') |      30 |
| ('environment', 'economy')    |      27 |
| ('politics', 'history')       |      26 |
| ('health', 'politics')        |      26 |
| ('culture', 'history')        |      23 |
| ('entertainment', 'fashion')  |      23 |
| ('religion', 'politics')      |      23 |
| ('entertainment', 'religion') |      22 |

## Confident mistakes
- true **fashion** -> predicted **entertainment** (0.99, igihe.com): Jay D wabaye Rudasumbwa wa Afurika yasezeye muri GodFather East Africa
- true **fashion** -> predicted **entertainment** (0.99, igihe.com): Eric Birasa agiye gutangiza uruganda nyuma y'imyaka umunani mu mideli
- true **politics** -> predicted **economy** (0.98, igihe.com): Ibigo byo mu Rwanda byatambukanye umucyo mu Imurikagurisha ry'Indabo mu Buholandi (Amafoto)
- true **politics** -> predicted **economy** (0.98, kigalipost.com): Perezida Kagame yatashye Icyambu gishya kidakora ku mazi k'i Masaka
- true **education** -> predicted **relationship** (0.99, kigalipost.com): Nkore iki ko nakundanye n'uwo dukorana?- Inama
- true **education** -> predicted **relationship** (0.98, kigalipost.com): Dore uko icyumweru kizakugendekera bitewe n'igihe wavukiye (Horoscope)
- true **environment** -> predicted **economy** (0.98, igihe.com): Byinshi kuri Banki yihariye izatera inkunga imishinga irengera ibidukikije mu Rwanda
- true **environment** -> predicted **economy** (0.97, igihe.com): Kayonza: Bralirwa yashyikirije abaturage amashanyarazi akomoka ku zuba inifatanya na bo gutera ibiti
- true **politics** -> predicted **history** (0.99, igihe.com): Hagaragajwe uko guhera mu 1959 hashinzwe ibinyamakuru byinshi byo kubiba amacakubiri mbere ya za 'Kangura'
- true **politics** -> predicted **history** (0.98, igihe.com): Imyaka 75 irashize Ingabo za Amerika zibohoye Umujyi wa Bastogne mu maboko y'Aba-Nazi (Amafoto)
- true **health** -> predicted **politics** (0.98, kigalipost.com): Kwizera uheruka kurushinga n'umugore umurusha imyaka 27, afunzwe akurikiranyweho gutera inda umwana
- true **health** -> predicted **politics** (0.96, kigalipost.com): Nyagatare: Umugabo yasanganywe imirambo y'abana bane mu nzu

## Confident correct predictions on rare topics
- **tourism** (0.98): Ibikorwa byo gusura ikiraro cyo mu bushorishori bwa Nyungwe byasubukuwe
- **environment** (0.93): Icyo REMA isaba abaturage mu bihe by'imvura n'ibiza
- **fashion** (0.98): Uyu munyamiderikazi yagurishije ubusugi bwe mu cyamunara ngo ashimishe nyina (AMAFOTO)
- **culture** (0.99): Inkomoko y'Insigamugani:Yatanze Rwantorero
