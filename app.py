import streamlit as st
import numpy as np
import pandas as pd

# =========================
# CONFIG UI
# =========================
st.set_page_config(
    page_title="Spam Email Classifier",
    layout="wide"
)

# CSS sederhana biar clean & modern
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }

    .center {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "login" not in st.session_state:
    st.session_state.login = False

if "hasil_batch" not in st.session_state:
    st.session_state.hasil_batch = None

if "trained" not in st.session_state:
    st.session_state.trained = False


# =========================
# MODEL JST
# =========================
if "w_ih" not in st.session_state:
    np.random.seed(42)
    st.session_state.w_ih = np.random.uniform(size=(5, 3))
    st.session_state.w_ho = np.random.uniform(size=(3, 1))
    st.session_state.b_h = np.random.uniform(size=(1, 3))
    st.session_state.b_o = np.random.uniform(size=(1, 1))


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# =========================
# DATA TRAINING (SESUAI LAPORAN)
# =========================
X_train = np.array([
    [1,0,0,0,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [0,0,0,1,0],
    [1,0,0,0,0],
    [0,0,0,0,1],
    [1,0,0,0,0],
    [0,0,1,0,0]
])

y_train = np.array([[1],[0],[1],[0],[1],[0],[1],[0]])


# =========================
# TRAIN MODEL
# =========================
def train_model():
    for _ in range(1000):

        hidden = sigmoid(np.dot(X_train, st.session_state.w_ih) + st.session_state.b_h)
        output = sigmoid(np.dot(hidden, st.session_state.w_ho) + st.session_state.b_o)

        error = y_train - output

        d_output = error * output * (1 - output)
        d_hidden = d_output.dot(st.session_state.w_ho.T) * hidden * (1 - hidden)

        st.session_state.w_ho += hidden.T.dot(d_output) * 0.2
        st.session_state.b_o += np.sum(d_output, axis=0, keepdims=True) * 0.2

        st.session_state.w_ih += X_train.T.dot(d_hidden) * 0.2
        st.session_state.b_h += np.sum(d_hidden, axis=0, keepdims=True) * 0.2


# =========================
# LOGIN PAGE (CENTERED, SIMPLE)
# =========================
if not st.session_state.login:

    st.markdown("<div class='title'>Email Spam Classifier</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if username == "admin" and password == "12345":
                    st.session_state.login = True
                    st.rerun()
                else:
                    st.error("Login gagal")

            st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# =========================
# DASHBOARD (NO SIDEBAR)
# =========================
st.markdown("<div class='title'>Dashboard Klasifikasi Email</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload Email", "Hasil Klasifikasi"])


# =========================
# TAB 1 - UPLOAD
# =========================
with tab1:

    st.markdown("### Upload Dataset Email")

    file = st.file_uploader("Upload CSV / TXT", type=["csv", "txt"])

    emails = []

    if file:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file, engine="python", on_bad_lines="skip")
            emails = df.iloc[:, 0].astype(str).tolist()
        else:
            content = file.read().decode("utf-8")
            emails = [x.strip() for x in content.split("\n") if x.strip()]

        st.markdown("### Preview Data")
        st.dataframe(pd.DataFrame(emails, columns=["Email"]))

        if st.button("Proses Klasifikasi"):

            if not st.session_state.trained:
                with st.spinner("Training model JST..."):
                    train_model()
                    st.session_state.trained = True

            results = []

            for email in emails:
                text = email.lower()

                fitur = np.array([[
                    1 if any(x in text for x in ["promo","gratis","diskon","hadiah"]) else 0,
                    1 if any(x in text for x in ["segera","klik","verifikasi"]) else 0,
                    1 if any(x in text for x in ["rapat","meeting","laporan"]) else 0,
                    1 if any(x in text for x in ["konfirmasi","jadwal"]) else 0,
                    1 if any(x in text for x in ["hai","teman","kabar"]) else 0
                ]])

                hidden = sigmoid(np.dot(fitur, st.session_state.w_ih) + st.session_state.b_h)
                output = sigmoid(np.dot(hidden, st.session_state.w_ho) + st.session_state.b_o)

                prob = float(output[0][0])
                label = "SPAM" if prob >= 0.5 else "NON-SPAM"

                results.append([email, prob, label])

            st.session_state.hasil_batch = results

            st.success("Klasifikasi selesai!")

            # AUTO PINDAH KE TAB HASIL (UX lebih bagus)
            st.info("Silakan buka tab 'Hasil Klasifikasi'")


# =========================
# TAB 2 - HASIL
# =========================
with tab2:

    st.markdown("### Hasil Klasifikasi")

    if st.session_state.hasil_batch is None:
        st.warning("Belum ada data diproses")
    else:
        df = pd.DataFrame(
            st.session_state.hasil_batch,
            columns=["Email", "Probabilitas", "Label"]
        )

        st.dataframe(df, use_container_width=True)

        spam = len(df[df["Label"] == "SPAM"])
        nonspam = len(df[df["Label"] == "NON-SPAM"])

        col1, col2 = st.columns(2)

        with col1:
            st.metric("SPAM", spam)

        with col2:
            st.metric("NON-SPAM", nonspam)

        st.success("Sistem JST Backpropagation berhasil melakukan klasifikasi email")

        