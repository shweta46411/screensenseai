
import streamlit as st
import pandas as pd
from datetime import datetime
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from dotenv import load_dotenv
import tensorflow as tf
# 🔐 OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY# Replace this with your actual API key

# Streamlit Config
st.set_page_config(page_title="ScreenSense AI", layout="wide")


st.markdown("""
<style>
/* 🌈 OUTER BACKGROUND (body) */
           body {
    background-color: #eaffea !important;
    background-image: linear-gradient(to bottom right, #eaffea, #f4fff4) !important;
    background-attachment: fixed !important;
    background-size: cover !important;
}

/* 🧱 OUTER APP VIEW CONTAINER */
[data-testid="stAppViewContainer"] {
    background: #eaffea !important;
    padding: 0 3rem !important;
}

/* 🧩 MAIN BLOCK CONTAINER - CARD-LIKE LOOK */
.block-container {
   background-color: #f6fbff !important; /* Inner white container */
    border-radius: 24px !important;
    padding: 2rem 2rem 4rem 2rem !important;
    max-width: 1200px !important;
    margin: 2rem auto !important;
    box-shadow: 0 8px 18px rgba(0,0,0,0.05) !important;
}

/* Headings */
h1, h2, h3 {
    color: #2a2a2a !important;
    font-family: 'Segoe UI Semibold', sans-serif !important;
    text-align: left !important;
}

/* Selectbox */
.stSelectbox > div {
    background-color: #fff !important;
    border: 2px solid #e0f0ff !important;
    border-radius: 14px !important;
    padding: 0.8rem !important;
    font-size: 16px !important;
    margin-bottom: 1rem !important;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    background-color: #e0eafc !important;
    padding: 0.5rem 1rem !important;
    border-radius: 14px !important;
    justify-content: left !important;
}
div[data-baseweb="tab"] {
    background-color: #ffffff !important;
    color: #333 !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    
}

/* Metric cards */
.stMetric {
    background-color: #ffffff !important;
    border: 1px solid #edf1f5 !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
    margin: 0.4rem !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.03) !important;
    text-align: center !important;
    font-size: 16px !important;
}

/* Section cards */
section[data-testid="stHorizontalBlock"] > div {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    border: 1px solid #e3e9f0;
}

/* GPT feedback cards */
.stAlert {
    background-color: #f4fbf9 !important;
    border-left: 4px solid #45c49c !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    font-size: 15px;
}

/* Hide footer */
footer, .css-1v3fvcr {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# 📊 Load Data
df = pd.read_csv("data/student_life_daily_features_with_names.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['uid', 'date'], inplace=True)
df.fillna(0, inplace=True)

# Get users
# users = df['uid'].unique()
uid_name_map = dict(zip(df['uid'], df['name']))
name_uid_map = dict(zip(df['name'], df['uid']))

# @st.cache_resource
# def load_sleep_model_and_assets():
#     model = tf.keras.models.load_model("data/final_hybrid_cnn_lstm_sleep_model.keras")
#     scaler = joblib.load("data/final_sleep_scaler.joblib")
#     with open("data/final_sleep_model_metadata.json") as f:
#         metadata = json.load(f)
#     return model, scaler, metadata

# sleep_model, sleep_scaler, sleep_meta = load_sleep_model_and_assets()

# # Load cleaned data (already preprocessed for sleep model)
# sleep_df = pd.read_csv("data/master_daily_data_cleaned_for_sleep_model.csv")
# sleep_df["date"] = pd.to_datetime(sleep_df["date"])
# sleep_df.sort_values(["uid", "date"], inplace=True)

# ---------------------- UI -----------------------
st.title("ScreenSense AI")

tab1, tab2 , tab3 = st.tabs(["📅 Daily View", "📈 Weekly Summary","Sleep Prediction"])

with tab1:

    valid_names = [name for name in name_uid_map.keys() if isinstance(name, str) and name.strip() != ""]
    selected_name = st.selectbox("👤 Select Student", sorted(valid_names))
    user = name_uid_map[selected_name]

    dates = df[df['uid'] == user]['date'].dt.date.unique()
    day = st.selectbox("📅 Select Day for Analysis", sorted(dates, reverse=True))

    row = df[(df['uid'] == user) & (df['date'].dt.date == day)].iloc[0]
    st.markdown(f"### 👩‍🎓 {selected_name} &nbsp;&nbsp;|&nbsp;&nbsp; 🗓️ {day}")

    # ---------- Metrics Section ----------
    st.markdown("### 📊 Daily Behavior & Mood Insights")

    # App Usage + Mood
    col1, col2, col3 = st.columns(3)
    col1.metric("📱 App Usage (min)", round(row['total_app_foreground_minutes'], 1), help="Daily screen time tracking")
    col1.progress(min(row['total_app_foreground_minutes'] / 600, 1.0))
    col2.metric("😊 Happy Intensity", round(row['avg_daily_happy_intensity'], 2), help="Positive emotion level")
    col2.progress(min(row['avg_daily_happy_intensity'] / 5, 1.0))
    col3.metric("😔 Sad Intensity", round(row['avg_daily_sad_intensity'], 2), help="Negative emotion level")
    col3.progress(min(row['avg_daily_sad_intensity'] / 5, 1.0))

    # Activity & Social
    col4, col5, col6 = st.columns(3)
    col4.metric("🏃 Running (min)", round(row['Running'], 1))
    col4.progress(min(row['Running'] / 100, 1.0))
    col5.metric("📚 Academic Time (min)", round(row['academic_study'], 1))
    col5.progress(min(row['academic_study'] / 600, 1.0))
    col6.metric("🎉 Social Time (min)", round(row['social_recreation'], 1))
    col6.progress(min(row['social_recreation'] / 300, 1.0))

    # Conversations
    col7, col8 = st.columns(2)
    col7.metric("🗣️ Conversations", int(row['number_of_conversations']))
    col7.progress(min(row['number_of_conversations'] / 20, 1.0))
    col8.metric("⏱️ Talk Duration (min)", round(row['total_conversation_duration_minutes'], 1))
    col8.progress(min(row['total_conversation_duration_minutes'] / 60, 1.0))

    # Mental Health
    col9, col10 = st.columns(2)
    col9.metric("😰 Stress Score", round(row['avg_daily_stress_score'], 2))
    col9.progress(min(row['avg_daily_stress_score'] / 5, 1.0))
    col10.metric("🧠 PHQ-9 Score", int(row['PHQ9_total_score']))
    col10.progress(min(row['PHQ9_total_score'] / 27, 1.0))

    # ---------- GPT Behavioral Feedback ----------
    st.markdown("### Insights")

    prompt = f"""
You are an empathetic wellness coach analyzing a student's digital behavior and mood on {day}.

Here is their data:
- Running: {row['Running']} min
- Academic Time: {row['academic_study']} min
- Social Time: {row['social_recreation']} min
- Conversations: {row['number_of_conversations']}
- Conversation Duration: {row['total_conversation_duration_minutes']} min
- App Usage: {row['total_app_foreground_minutes']} min
- Distinct Apps Used: {row['distinct_apps_used']}
- Happy Intensity: {row['avg_daily_happy_intensity']}
- Sad Intensity: {row['avg_daily_sad_intensity']}
- Stress Score: {row['avg_daily_stress_score']}
- PHQ9 Score: {row['PHQ9_total_score']}

Please provide a supportive 2-part response:
1. **🧠 Behavioral Analysis** (2–3 lines): Explain patterns, correlations, or anomalies in the student's day. Focus on balance, mood, and engagement.
2. **🌿 Wellness Suggestion** (1 clear tip): Offer one specific, actionable suggestion to help improve their well-being based on the data.

Make the tone warm, human, and insightful. Use markdown formatting exactly like this:
**🧠 Behavioral Analysis:**
<your analysis here>

**🌿 Wellness Suggestion:**
<your tip here>
"""

    with st.spinner("Analyzing behavior..."):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
    gpt_output = response['choices'][0]['message']['content']
    st.markdown(f"#### 🔎 Behavioral Analysis")
    st.info(gpt_output.split("**🌿")[0].replace("**🧠", "").strip())
    st.markdown(f"#### 🌿 Wellness Suggestion")
    st.success("🌱 " + gpt_output.split("**🌿")[-1].strip())


# ---------------------- WEEKLY ----------------------
with tab2:
    

    # Filter out empty or invalid names
    valid_names = [name for name in name_uid_map.keys() if isinstance(name, str) and name.strip() != ""]

    # Student selection
    selected_name_weekly = st.selectbox("Select Student", sorted(valid_names), key="weekly_student")
    selected_uid_weekly = name_uid_map[selected_name_weekly]

    # Filter data for selected user
    user_df = df[df["uid"] == selected_uid_weekly].copy()
    user_df["week"] = user_df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    st.header(f"📊 Weekly Trends for {selected_name_weekly}")

    # Aggregate by week
    weekly_df = user_df.groupby("week").agg({
        "avg_daily_happy_intensity": "mean",
        "avg_daily_sad_intensity": "mean",
        "total_app_foreground_minutes": "sum"
    }).reset_index()

    if weekly_df.empty:
        st.warning("No weekly data available for this student.")
    else:
        # Mood Trend Chart
        st.markdown("### 😊 Mood Trends Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=weekly_df, x="week", y="avg_daily_happy_intensity", label="Happy", marker="o")
        sns.lineplot(data=weekly_df, x="week", y="avg_daily_sad_intensity", label="Sad", marker="o")
        ax.set_ylabel("Mood Intensity")
        ax.set_xlabel("Week")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # App Usage Chart
        st.markdown("### 📱 App Usage Over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=weekly_df, x="week", y="total_app_foreground_minutes", color="skyblue")
        ax2.set_ylabel("App Minutes")
        ax2.set_xlabel("Week")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # GPT Weekly Insight
        latest = weekly_df.sort_values("week").iloc[-1]
        summary_prompt = f"""
        For the week starting {latest['week'].strftime('%Y-%m-%d')}, the student had:
        - Happy intensity: {latest['avg_daily_happy_intensity']:.2f}
        - Sad intensity: {latest['avg_daily_sad_intensity']:.2f}
        - App usage: {latest['total_app_foreground_minutes']:.0f} minutes

        Write a 2-line insight and one improvement tip for wellbeing.
        """
        with st.spinner("🧠 Generating  Weekly Insight..."):
            response2 = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}]
            )
            st.markdown("### 🤖  Weekly Feedback")
            st.info(response2['choices'][0]['message']['content'])
# with tab3:
#     st.header("🛌 Sleep Prediction")

#     # Load assets (cached for performance)
#     @st.cache_resource
#     def load_sleep_assets():
#         model = tf.keras.models.load_model("data/final_hybrid_cnn_lstm_sleep_model.keras")
#         scaler = joblib.load("data/final_sleep_scaler.joblib")
#         with open("data/final_sleep_model_metadata.json") as f:
#             metadata = json.load(f)
#         data = pd.read_csv("data/master_daily_data_with_names.csv", parse_dates=["date"])
#         return model, scaler, metadata, data

#     model, scaler, meta, sleep_df = load_sleep_assets()

#     # Extract metadata
#     feature_cols = meta["feature_columns"]
#     seq_len = meta["sequence_length"]
#     threshold = meta["optimal_threshold"]
#     labels = meta["target_names"]

#     # ✅ Filter valid names safely
#     valid_names_sleep = sorted(
#         [name for name in df["name"].dropna().unique() if isinstance(name, str) and name.strip() != ""]
#     )
#     selected_name_sleep = st.selectbox("Select Student", valid_names_sleep, key="sleep_user")
#     selected_uid_sleep = name_uid_map[selected_name_sleep]

#     # Filter and prepare student data
#     user_sleep_df = sleep_df[sleep_df["uid"] == selected_uid_sleep].copy()
#     user_sleep_df = user_sleep_df.sort_values("date").reset_index(drop=True)

#     # Generate list of valid prediction dates
#     valid_dates = user_sleep_df["date"].iloc[seq_len:].dt.date.unique()
#     selected_date = st.selectbox("Select Date to Predict Sleep Quality", valid_dates, key="sleep_date")

#     # Get the 7-day sequence before selected date
#     try:
#         end_idx = user_sleep_df[user_sleep_df["date"].dt.date == selected_date].index[0]
#         start_idx = end_idx - seq_len
#         if start_idx < 0:
#             raise IndexError("Not enough prior data.")
#         seq_df = user_sleep_df.iloc[start_idx:end_idx][feature_cols]
#     except Exception as e:
#         st.error(f"Not enough data to make a prediction for this date. ({e})")
#         st.stop()

#     # Validate sequence length
#     if seq_df.shape[0] != seq_len:
#         st.error("Not enough prior data to make a prediction.")
#     else:
#         # Scale and reshape
#         input_data = scaler.transform(seq_df)
#         input_data = input_data.reshape(1, seq_len, len(feature_cols))

#         # Predict
#         pred_proba = model.predict(input_data)[0][0]
#         prediction = int(pred_proba >= threshold)
#         label = labels[prediction]

#         # Display
#         st.subheader(f"🛏️ Predicted Sleep Quality for {selected_name_sleep} on {selected_date}")
#         st.success(f"**{label}** (Confidence: `{pred_proba:.2f}`)")

#         # Simple explanation
#         st.markdown("#### 📘 Explanation")
#         if prediction == 1:
#             st.info("Good sleep predicted! The student shows healthy behavioral patterns over the past week.")
#         else:
#             st.warning("Poor sleep predicted. Encourage better evening routines or less screen time before bed.")

#         # ----------------- GPT Feedback ------------------
#         st.markdown("#### 🤖 Personalized Sleep Insight")

#         behavior_summary = seq_df.mean().to_dict()
#         summary_text = "\n".join([f"- {k.replace('_', ' ').title()}: {round(v, 2)}" for k, v in behavior_summary.items()])

#         sleep_feedback_prompt = f"""
# You are a helpful AI coach. A student's sleep quality was predicted as **{label}** on {selected_date} using 7 days of behavior data.

# Here is the average of their past 7 days' data:
# {summary_text}

# Please provide:
# 1. **🧠 Sleep Insight**: In 2–3 lines, explain patterns in their behavior that likely led to this sleep prediction.
# 2. **🌿 Wellness Tip**: One actionable recommendation for the student to improve or maintain sleep quality.

# Use markdown formatting exactly like this:
# **🧠 Sleep Insight:**
# <your insight here>

# **🌿 Wellness Tip:**
# <your tip here>
# """

#         with st.spinner("🧠 Generating personalized feedback..."):
#             response = openai.ChatCompletion.create(
#                 model="gpt-4",
#                 messages=[{"role": "user", "content": sleep_feedback_prompt}]
#             )
#             gpt_sleep_output = response['choices'][0]['message']['content']
#             st.info(gpt_sleep_output)
# with tab3:
#     st.markdown("## 🛌 Predict Upcoming Sleep Quality")
#     st.markdown("""
#     This feature uses a student's **last 7 days of behavior** to predict whether they will have a **Good Sleep** or **Poor Sleep** on the selected date.
    
#     The prediction is made by a **CNN-LSTM model**, trained on time series patterns from student behavior like:
#     - 📱 App Usage
#     - 🏃 Physical Activity
#     - 🗣️ Conversations
#     - 🧠 Mental Health Scores
#     """)

#     @st.cache_resource
#     def load_sleep_assets():
#         model = tf.keras.models.load_model("data/final_hybrid_cnn_lstm_sleep_model.keras")
#         scaler = joblib.load("data/final_sleep_scaler.joblib")
#         with open("data/final_sleep_model_metadata.json") as f:
#             metadata = json.load(f)
#         data = pd.read_csv("data/master_daily_data_with_names.csv", parse_dates=["date"])
#         return model, scaler, metadata, data

#     model, scaler, meta, sleep_df = load_sleep_assets()
#     feature_cols = meta["feature_columns"]
#     seq_len = meta["sequence_length"]
#     threshold = meta["optimal_threshold"]
#     labels = meta["target_names"]

#     valid_names = sorted([n for n in sleep_df["name"].dropna().unique() if isinstance(n, str) and n.strip()])
#     selected_name = st.selectbox("👤 Select Student", valid_names, key="student_selector_tab3")
#     selected_uid = sleep_df[sleep_df["name"] == selected_name]["uid"].iloc[0]

#     user_df = sleep_df[sleep_df["uid"] == selected_uid].sort_values("date").reset_index(drop=True)
#     valid_dates = user_df["date"].iloc[seq_len:].dt.date.unique()
#     selected_date = st.selectbox("📅 Select Date to Predict Sleep", valid_dates, key="date_selector_tab3")

#     try:
#         end_idx = user_df[user_df["date"].dt.date == selected_date].index[0]
#         start_idx = end_idx - seq_len
#         if start_idx < 0:
#             raise IndexError("Not enough prior data.")
#         seq_df = user_df.iloc[start_idx:end_idx][feature_cols]
#     except Exception as e:
#         st.error(f"❗ Not enough data to make a prediction for this date. ({e})")
#         st.stop()

#     if seq_df.shape[0] != seq_len:
#         st.warning("⚠️ Not enough prior data for a full 7-day sequence.")
#         st.stop()

#     input_data = scaler.transform(seq_df).reshape(1, seq_len, len(feature_cols))
#     pred_proba = model.predict(input_data)[0][0]
#     prediction = int(pred_proba >= threshold)
#     label = labels[prediction]
#     color = "#3cb371" if prediction == 1 else "#ff6b6b"

#     st.markdown(f"""
#     <div style="padding: 1rem; border-radius: 12px; background-color: {color}; color: white; font-size: 1.2rem">
#         <b>Predicted Sleep Quality:</b> {label}  
#         <br>
#         <b>Confidence:</b> {pred_proba:.2f}
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("### 📊 Last 7 Days: Behavior Trends")
#     st.markdown("Behavioral indicators used to predict sleep quality.")
#     fig, ax = plt.subplots(figsize=(10, 3))
#     cols_to_plot = [col for col in ["Running", "academic_study", "total_app_foreground_minutes"] if col in seq_df.columns]
#     if cols_to_plot:
#         seq_df[cols_to_plot].plot(ax=ax)
#         ax.set_ylabel("Minutes / Intensity")
#         ax.set_title("Running | Academic | App Usage (7-day window)")
#         plt.xticks(rotation=45)
#         st.pyplot(fig)
#     else:
#         st.info("No matching columns found for plotting trends.")

#     st.markdown("### 🤖 Personalized Insight from GPT")
#     behavior_summary = seq_df.mean().to_dict()
#     summary_text = "\n".join([f"- {k.replace('_', ' ').title()}: {round(v, 2)}" for k, v in behavior_summary.items()])

#     prompt = f"""
# You are a helpful AI coach. A student's sleep quality was predicted as **{label}** on {selected_date} using 7 days of behavior data.

# Here is the average of their past 7 days' data:
# {summary_text}

# Please provide:
# 1. **🧠 Sleep Insight**: In 2–3 lines, explain patterns in their behavior that likely led to this sleep prediction.
# 2. **🌿 Wellness Tip**: One actionable recommendation for the student to improve or maintain sleep quality.

# Use markdown formatting exactly like this:
# **🧠 Sleep Insight:**
# <your insight here>

# **🌿 Wellness Tip:**
# <your tip here>
# """

#     with st.spinner("Generating AI-generated feedback..."):
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         gpt_response = response['choices'][0]['message']['content']
#         st.info(gpt_response)

with tab3:
    st.markdown("## 🛌 Predict Upcoming Sleep Quality")
    st.markdown("""
    This feature uses a student's **last 7 days of behavior** to predict whether they will have a **Good Sleep** or **Poor Sleep** on the selected date.
    
    The prediction is made by a **CNN-LSTM model**, trained on time series patterns from student behavior like:
    - 📱 App Usage
    - 🏃 Physical Activity
    - 🗣️ Conversations
    - 🧠 Mental Health Scores
    """)

    @st.cache_resource
    def load_sleep_assets():
        model = tf.keras.models.load_model("data/final_hybrid_cnn_lstm_sleep_model.keras")
        scaler = joblib.load("data/final_sleep_scaler.joblib")
        with open("data/final_sleep_model_metadata.json") as f:
            metadata = json.load(f)
        data = pd.read_csv("data/master_daily_data_with_names.csv", parse_dates=["date"])
        return model, scaler, metadata, data

    model, scaler, meta, sleep_df = load_sleep_assets()
    feature_cols = meta["feature_columns"]
    seq_len = meta["sequence_length"]
    threshold = meta["optimal_threshold"]
    labels = meta["target_names"]

    valid_names = sorted([n for n in sleep_df["name"].dropna().unique() if isinstance(n, str) and n.strip()])
    selected_name = st.selectbox("👤 Select Student", valid_names, key="student_selector_tab3")
    selected_uid = sleep_df[sleep_df["name"] == selected_name]["uid"].iloc[0]

    user_df = sleep_df[sleep_df["uid"] == selected_uid].sort_values("date").reset_index(drop=True)
    valid_dates = user_df["date"].iloc[seq_len:].dt.date.unique()
    selected_date = st.selectbox("📅 Select Date to Predict Sleep", valid_dates, key="date_selector_tab3")

    try:
        end_idx = user_df[user_df["date"].dt.date == selected_date].index[0]
        start_idx = end_idx - seq_len
        if start_idx < 0:
            raise IndexError("Not enough prior data.")
        seq_df = user_df.iloc[start_idx:end_idx][feature_cols]
    except Exception as e:
        st.error(f"❗ Not enough data to make a prediction for this date. ({e})")
        st.stop()

    if seq_df.shape[0] != seq_len:
        st.warning("⚠️ Not enough prior data for a full 7-day sequence.")
        st.stop()

    try:
        input_data = scaler.transform(seq_df).reshape(1, seq_len, len(feature_cols))
        pred_proba = model.predict(input_data, verbose=0)[0][0]
        prediction = int(pred_proba >= threshold)
        label = labels[prediction]
        color = "#3cb371" if prediction == 1 else "#ff6b6b"
    except Exception as e:
        st.error(f"❗ Model prediction failed. Check model input shape or TF version. ({e})")
        st.stop()

    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 12px; background-color: {color}; color: white; font-size: 1.2rem">
        <b>Predicted Sleep Quality:</b> {label}  
        <br>
        <b>Confidence:</b> {pred_proba:.2f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Last 7 Days: Behavior Trends")
    fig, ax = plt.subplots(figsize=(10, 3))
    cols_to_plot = [col for col in ["Running", "academic_study", "total_app_foreground_minutes"] if col in seq_df.columns]
    if cols_to_plot:
        seq_df[cols_to_plot].plot(ax=ax)
        ax.set_ylabel("Minutes / Intensity")
        ax.set_title("Running | Academic | App Usage (7-day window)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No matching columns found for plotting trends.")

    st.markdown("### 🤖 Personalized Insight from GPT")
    behavior_summary = seq_df.mean().to_dict()
    summary_text = "\n".join([f"- {k.replace('_', ' ').title()}: {round(v, 2)}" for k, v in behavior_summary.items()])

    prompt = f"""
You are a helpful AI coach. A student's sleep quality was predicted as **{label}** on {selected_date} using 7 days of behavior data.

Here is the average of their past 7 days' data:
{summary_text}

Please provide:
1. **🧠 Sleep Insight**: In 2–3 lines, explain patterns in their behavior that likely led to this sleep prediction.
2. **🌿 Wellness Tip**: One actionable recommendation for the student to improve or maintain sleep quality.

Use markdown formatting exactly like this:
**🧠 Sleep Insight:**
<your insight here>

**🌿 Wellness Tip:**
<your tip here>
"""

    with st.spinner("Generating AI-generated feedback..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            gpt_response = response['choices'][0]['message']['content']
            st.info(gpt_response)
        except Exception as e:
            st.error(f"❗ Failed to fetch GPT feedback. ({e})")
