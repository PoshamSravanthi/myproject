import streamlit as st
from app import run_multi_agent_workflow, WorkflowInput

st.set_page_config(page_title="🧠 Deep Think", layout="centered")

st.title("🧠 Deep Think")
st.write("Enter a topic below and let the AI agents research, summarize, and write for you.")

query = st.text_input("Research Topic:", placeholder="e.g., Latest AI trends in India")
start = st.button("Start Research")

if start:
    if not query.strip():
        st.warning("Please enter a topic before starting.")
    else:
        with st.spinner("Agents are researching... Please wait..."):
            result = run_multi_agent_workflow(
                WorkflowInput(query=query)
            )

        st.write("---")

        tab1, tab2, tab3 = st.tabs(["🔎 Research Data", "📑 Summary", "✉ Email"])

        # TAB 1 - Research Data
        with tab1:
            st.subheader("Raw Research Data")
            st.text(result.get("raw_research", "No research data found."))

        # TAB 2 - Summary
        with tab2:
            summary = result.get("summary", {})

            st.subheader("Executive Summary")
            st.write(summary.get("executive_summary", "No summary found."))

            st.subheader("Action Items")
            action_items = summary.get("action_items", [])
            if action_items:
                for item in action_items:
                    st.write(f"- {item}")
            else:
                st.write("No action items found.")

        # TAB 3 - Email
        with tab3:
            st.subheader("Generated Email")
            st.text_area(
                label="",
                value=result.get("final_email", "No email content found."),
                height=250
            )
