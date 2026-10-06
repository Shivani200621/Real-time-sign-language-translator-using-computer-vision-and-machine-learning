import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="SignEase - Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOGIN ----------------

USERNAME = "student"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.markdown(
        '<div class="main-title">🤟 SignEase</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Real-Time Sign Language Translator</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.subheader("🔐 Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button(
            "Login",
            use_container_width=True
        ):

            if username == USERNAME and password == PASSWORD:

                st.session_state.logged_in = True
                st.rerun()

            else:

                st.error(
                    "Invalid username or password"
                )

    st.info(
        "Demo Login → Username: student | Password: 1234"
    )


# ---------------- MAIN APP ----------------

else:

    # Load trained model

    with open("sign_model.pkl", "rb") as f:
        model = pickle.load(f)


    # ---------------- SIDEBAR ----------------

    st.sidebar.title("🤟 SignEase")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "📷 Translate Sign",
            "📖 Supported Signs",
            "ℹ️ About Project"
        ]
    )

    if st.sidebar.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.rerun()


    # ---------------- HOME ----------------

    if page == "🏠 Home":

        st.markdown(
            '<div class="main-title">🤟 SignEase</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Real-Time Sign Language Translator'
            '</div>',
            unsafe_allow_html=True
        )

        st.success(
            "Welcome! Convert hand signs into understandable text."
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### 📷 Camera")

            st.write(
                "Capture your hand sign using your camera."
            )

        with col2:

            st.markdown("### 🧠 AI Detection")

            st.write(
                "The trained machine learning model analyzes your hand landmarks."
            )

        with col3:

            st.markdown("### 💬 Output")

            st.write(
                "The recognized sign is displayed as text."
            )

        st.divider()

        st.subheader("🚀 How to Use")

        st.write("1. Go to **Translate Sign**.")
        st.write("2. Allow camera access.")
        st.write("3. Show your hand clearly.")
        st.write("4. Capture the image.")
        st.write("5. The AI model predicts the sign.")

        st.divider()

        st.subheader("✨ Current Supported Signs")

        signs = ["👋 Hello", "👍 Yes", "👎 No", "✋ Stop", "🙏 Thank You"]

        cols = st.columns(5)

        for i, sign in enumerate(signs):

            with cols[i]:

                st.info(sign)


    # ---------------- TRANSLATOR ----------------

    elif page == "📷 Translate Sign":

        st.title("📷 Sign Language Translator")

        st.write(
            "Show one hand clearly in front of the camera."
        )

        st.info(
            "💡 Tip: Keep your hand inside the camera frame "
            "and use good lighting."
        )

        picture = st.camera_input(
            "Take a picture of your sign"
        )

        if picture is not None:

            file_bytes = np.asarray(
                bytearray(picture.getvalue()),
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )


            # MediaPipe Hand Detection

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(

                base_options=BaseOptions(
                    model_asset_path="hand_landmarker.task"
                ),

                running_mode=VisionRunningMode.IMAGE,

                num_hands=1
            )

            detector = HandLandmarker.create_from_options(
                options
            )


            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=rgb
            )


            result = detector.detect(
                mp_image
            )


            # ---------------- PREDICTION ----------------

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                data = []

                for landmark in hand:

                    data.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])


                data = np.array(
                    data
                ).reshape(1, -1)


                prediction = model.predict(
                    data
                )

                sign = prediction[0]


                st.success(
                    "✅ Hand detected successfully!"
                )


                st.markdown(
                    f'<div class="result">'
                    f'🤟 Detected Sign<br><br>'
                    f'{sign.upper()}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.write("")

                st.subheader("💬 Translation")

                st.write(
                    f"The detected sign represents: **{sign}**"
                )


            else:

                st.warning(
                    "⚠️ No hand detected. "
                    "Please show your hand clearly."
                )


    # ---------------- SUPPORTED SIGNS ----------------

    elif page == "📖 Supported Signs":

        st.title("📖 Supported Signs")

        st.write(
            "The current trained model recognizes these signs:"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 👋 Hello")

            st.write(
                "Used as a greeting."
            )

            st.markdown("### 👍 Yes")

            st.write(
                "Used to indicate agreement."
            )

            st.markdown("### 👎 No")

            st.write(
                "Used to indicate disagreement."
            )

        with col2:

            st.markdown("### ✋ Stop")

            st.write(
                "Used to ask someone to stop."
            )

            st.markdown("### 🙏 Thank You")

            st.write(
                "Used to express gratitude."
            )


        st.info(
            "More signs can be added later by collecting "
            "training samples and retraining the model."
        )


    # ---------------- ABOUT ----------------

    elif page == "ℹ️ About Project":

        st.title("ℹ️ About the Project")

        st.subheader(
            "🤟 Real-Time Sign Language Translator"
        )

        st.write(
            "This project uses computer vision and machine "
            "learning to recognize hand signs from a camera image."
        )

        st.subheader("🔧 Technologies Used")

        st.write("• Python")
        st.write("• Streamlit")
        st.write("• OpenCV")
        st.write("• MediaPipe")
        st.write("• Scikit-learn")
        st.write("• Machine Learning")

        st.subheader("🎯 Objective")

        st.write(
            "The main objective is to provide a simple system "
            "that converts selected hand signs into readable text."
        )

        st.subheader("🌍 Future Scope")

        st.write(
            "The system can be extended with more signs, "
            "voice output, sentence formation and real-time video recognition."
        )