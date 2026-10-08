# Streamlit app: loads the fine-tuned model from the HF Hub and classifies pasted articles.
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinnews.text import normalize  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Samkwizera/kinnews-topic-classifier")
MAX_LENGTH = 512
LOW_CONFIDENCE = 0.5

KIN_NAMES = {
    "politics": "Politiki", "sport": "Imikino", "economy": "Ubukungu", "health": "Ubuzima",
    "entertainment": "Imyidagaduro", "history": "Amateka", "technology": "Ikoranabuhanga",
    "tourism": "Ubukerarugendo", "culture": "Umuco", "fashion": "Imideri", "religion": "Iyobokamana",
    "environment": "Ibidukikije", "education": "Uburezi", "relationship": "Urukundo",
}


# cached so the model is downloaded and loaded once, not on every click
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    return tokenizer, model


@torch.no_grad()
def classify(title, body):
    tokenizer, model = load_model()
    # same normalisation as training, otherwise apostrophes etc. tokenise differently
    text = normalize(f"{title}. {body}" if title.strip() else body)
    n_tokens = len(tokenizer(text)["input_ids"])
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    probs = torch.softmax(model(**enc).logits[0], dim=-1)
    ranked = sorted(((model.config.id2label[i], float(p)) for i, p in enumerate(probs)), key=lambda x: -x[1])
    return ranked, n_tokens


# real articles from the KINNEWS test set: three the model gets right, two it gets wrong
EXAMPLES = [
    ('Euro 2016: Portugal yamaze kwicara muri 1/2 kirangiza',
     "Ikipe y'igihugu ya Portugal ibaye ikipe yambere ibimburiye izindi kugera muri kimwe cya kabiri kirangiza, nyuma yo gutsinda Poland kuri penaliti 5-3. Robert Lewandowski wa Poland niwe wafunguye amazamu, umukino umaze amasegonda 100 utangiye, igitego cyaje kwishyurwa ku munota wa 33 n'umukinnyi Renato Sanches wa Portugal. Cristiano Ronaldo yagiye ahusha uburyo bwiza bwari kubyazwamo ibitego, umukino waje kurangira amakipe yombi anganya igitego 1-1. Hitabazwa iminota 30 y'inyongera ngo haboneke ikipe igera muri kimwe cya kabiri kirangiza Iminota 30 yarangiye nta kipe ibashije kureba mu izamu ry'indi, amakipe yombi yitabaza penaliti. Portugal yinjiza, 5 kuri 3 za Poland, ku ruhande rwa Portugal zatewe na Cristiano Ronaldo, Renato Sanches, Joao Moutinho, Nani na Ricardo Quaresma winjije iya nyuma yabahaye itsinzi. Umukinnyi warase penaliti ku ruhande rwa Poland ni Jakub Blaszczykowsk, abandi bari bazinjize ni Robert Lewandowski, Arkadiusz Milik na Kamil Glik. Portugal ikaba itegereje kuzakina n'ikipe izatsinda hagati ya Wales na Belgium ziza gukina kuri uyu wa gatanu tariki ya 1 Nyakanga 2016. Yanditswe na Ubwanditsi/Muhabura.rw"),
    ("Inyamaswa ziri muri Pariki y'Akagera zageze ku bihumbi 13",
     "Kubarura inyamaswa ziri muri iyi pariki byatangiye gukorwa mu 2010 mu ibarura rikorwa buri myaka ibiri. Uyu mwaka ryakozwe hifashijwe indege ya kajugujugu yo mu bwoko bwa AS350 B3 yagenzuye ibice byose bya Pariki. Imibare itangazwa na Pariki y'Akagera igaragaza ko raporo y'ibarura yerekanye inyamaswa zose ziri muri iyi pariki ari 13500 aho bigaragara ko ziyongereye zikava ku 12000 zabarurwaga mu 2017. Iyi pariki ni imwe mu zisurwa cyane mu Rwanda aho umubare munini wa ba mukerarugendo bayigana ari abanyarwanda bihariye 50% by'abayisura bose. Mu mwaka ushize yasuwe n'abantu 44000. Aba bayisuye bayinjirije amadorali miliyoni ebyiri ni ukuvuga amafaranga y'u Rwanda arenga miliyari imwe na miliyoni 800. Parike y'Igihugu y'Akagera iherereye mu Ntara y'Iburasirazuba yashinzwe mu 1934. Uretse kuba ari ahantu hagizwe n'uduce tw'imirambi habereye abahasura, harimo n'inyamanswa zikundwa na ba mukerarugendo nk'Imparage, Udusumbashyamba, Inzovu, Intare, Ingona n'Imvubu biri mu Kiyaga cya Ihema n'andi moko y'inyoni. Iyi pariki ifite ubuso bungana na kilometero kare 1122, irimo inyamaswa zizwi nka 'Big Five' ari zo; Intare, Inkuru, Inzovu, Ingwe n'Imbogo."),
    ('Pasiteri wari ukuriye Ivugabutumwa muri ADEPR yirukanwe mu itorero',
     'Itorero rya Pantekote mu Rwanda ryirukanye Pasiteri Zigirinshuti Michel, uzwi cyane muri ADEPR, wari ukuriye ivugabutumwa, wamamaye kubera imvugo zidasanzwe akoresha abwiriza. Iki cyemezo ADEPR yagifashe nyuma y\'uko Pasiteri Zigirinshuti amaze igihe kirenga ukwezi yarataye akazi ke, adasabye uruhushya nk\'uko Umuvugizi wa ADEPR, Rev. Karuranga yabitangaje Yagize ati "Pasiteri Zigirinshuti twamwirukanye kubera ko yataye akazi. Amakuru mfite ni uko ngo yari yaragiye mu bihugu byo hanze. Ntiyigeze atumenyesha n\'umuyobozi we wa hafi muri serivisi akoramo ntiyigeze amumenyesha kandi ntiyari mu kiruhuko." Rev. Karuranga yakomeje avuga ko kwirukana Zigirinshuti ADEPR nta tegeko yishe ngo kuko yarengeje iminsi umunani umukozi yihanangirizwa cyangwa agasabwa ibisobanuro. Karuranga ati "Nta tegeko twishe kuko yarengeje iminsi umunani itegeko rivuga ko umukozi yihanangirizwa. We yamaze ukwezi kurenga." Pasiteri Zigirinshuti Michel yaherukaga mu ivugabutumwa muri Afurika y\'Epfo mu giterane cyateguwe n\'Itorero Fullness of God International Ministries rifatanyije na Parani Pentecostal Church Ministry. Iki giterane cyari gifite insangamatsiko iboneka muri Yeremiya 31:4 handitse hati "Nzongera kukubaka wa mwari wa Isiraheli we." Twagerageje kuvugisha Zigirinshuti ariko ntabwo twamufatishije kuri telefone ye igendanwa. Chief editor'),
    ("U Rwanda rwahawe miliyoni 32.8$ zo guhangana n'imihindagurikire y'ibihe",
     "Uyu mushinga uzabarizwa mu kigega gitera inkunga imishinga yita ku bidukikije mu Rwanda (Fonerwa), ugamije gutunganya igishanga cya Muvumba, kongera ubushobozi bw'abaturage mu gutunganya amashyamba ndetse no gufasha abahinzi bato gukora ubuhinzi buhangana n'ihindagurika ry'ibihe. Uzakorera mu mirenge icyenda y'Akarere ka Gicumbi ari yo Kaniga, Rubaya, Cyumba, Rushaki, Shangasha, Mukarange, Manyagiro, Byumba na Bwisige. Uzafasha abatuye ahantu haba inkangu n'imyuzure gutura mu buryo buhangana n'imihindagurikire y'ibihe kandi wite ku baturage bakennye cyane badafite ubushobozi. Abaturage bazigishwa uburyo burambye bwo kubungabunga amashyamba, imiturire igezweho, uburyo bwo gutunganya ibishanga n'ibindi, hanyuma buzajye no gukoreshwa mu gihugu hose. Biteganyijwe ko mu myaka itandatu uyu mushinga uzafasha abaturage kugabanya imyuka ihumanya (CO2), ingana na toni 273,720. Uzagera ku baturage 150,000 ugirire akamaro abarenga 380,000 muri rusange. Uretse izi miliyoni 32.8 z'amadolari, Ikigega cyo kubungabunga ibidukikije mu Rwanda (RGF), kizatangamo ibihumbi 147 z'amadolari ndetse n'Akarere ka Gicumbi gatange ibihumbi 107 $ naho umuryango 'The Wood Foundation' utange 105,964 $."),
    ("Ibisobanuro by'izina Briella n'imico ikunze kuranga abaryitwa",
     'Briella ni izina ry\'abakobwa rikomoka ku rurimi rw\'Igiheburayo risobanura "Imana niyo mbaraga zanjye" Ba Briella ni abantu usanga badakunze kwigaragaza cyane akenshi bagamije kwihesha agaciro. Ba Briella ntabwo ari abantu umuntu ashobora gusobanukirwa, byose biterwa n\'uko yaramutse. Ntabwo bakunze guha icyizere ubonetse wese. Ntibishimira akarengane. Bakunda umuryango wabo kandi baba ababyeyi beza cyane, ntabwo babasha kwihanganira kubaho badafite abana. Ni abajyanama beza kandi babasha kuba abahuza cyangwa abunzi beza. Mu rukundo, Briella yifuza umugabo ugira ubuntu kandi umugaragariza urukundo amufitiye, ariko akamenya kumuhakanira mu gihe biri ngombwa. Aba akeneye gukunda kugira ngo amererwe neza kandi ntaba yifuza na gato kubaho wenyine. Nubwo aba akeneye kugira uwo yita umukunzi ariko aba yifuza ko amuha umwanya we, kandi ntiyihanganira na gato umuntu ufuha. Akeneye umugabo utuma yumva atekanye haba ku mubiri ndetse no mu butunzi.'),
]

st.set_page_config(page_title="KINNEWS Topic Classifier", page_icon="📰")
st.title("Kinyarwanda News Topic Classifier")
st.write(
    "Paste a Kinyarwanda news article and the model predicts its topic among 14 categories. "
    f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (AfriBERTa fine-tuned on KINNEWS). "
    "Code: [GitHub](https://github.com/Samkwizera/NLP-application)."
)

if "title" not in st.session_state:
    st.session_state.title, st.session_state.body = "", ""

choice = st.selectbox("Try an example from the test set", ["-"] + [t for t, _ in EXAMPLES])
if choice != "-" and choice != st.session_state.get("last_choice"):
    st.session_state.title, st.session_state.body = choice, dict(EXAMPLES)[choice]
st.session_state.last_choice = choice

title = st.text_input("Title (Umutwe)", key="title")
body = st.text_area("Article text (Inkuru)", key="body", height=250)

if st.button("Classify", type="primary"):
    if len(f"{title} {body}".split()) < 3:
        st.warning("Please enter at least a sentence of Kinyarwanda text.")
    else:
        with st.spinner("Loading model and classifying..."):
            ranked, n_tokens = classify(title, body)
        label, conf = ranked[0]
        st.subheader(f"{label} ({KIN_NAMES[label]}) - {conf:.0%}")
        if conf < LOW_CONFIDENCE:
            st.warning(f"Low confidence ({conf:.0%}): the article may cover several topics, "
                       "or a topic that was rare in the training data.")
        top5 = pd.DataFrame({"probability": [p for _, p in ranked[:5]]},
                            index=[f"{l} ({KIN_NAMES[l]})" for l, _ in ranked[:5]])
        st.bar_chart(top5, horizontal=True, sort=False)
        note = f"Input: {n_tokens} subword tokens"
        if n_tokens > MAX_LENGTH:
            note += f" (only the first {MAX_LENGTH} were used)"
        st.caption(note)
