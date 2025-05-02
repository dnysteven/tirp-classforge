# src/streamlit_app.py
from data_prep import load_and_scale, simulate_graph
from ga_engine import setup_deap, run_ga
import streamlit as st
import pandas as pd

def main():
    st.title("ClassForge: GA-Powered Classroom Allocation")

    # 1. Sidebar controls
    K      = st.sidebar.number_input("Number of classes", 2, 10, 4)
    w_acad = st.sidebar.slider("Academic balance weight", 0.0, 1.0, 0.33)
    w_fri  = st.sidebar.slider("Friendship retention weight", 0.0, 1.0, 0.33)
    w_neg  = st.sidebar.slider("Disrespect penalty weight", 0.0, 1.0, 0.33)
    pop    = st.sidebar.slider("Population size", 2, 100, 20)
    gens   = st.sidebar.slider("Generations", 1, 50, 5)

    # 2. Load & prepare data
    df, feature_cols = load_and_scale()

    # df_scaled, feature_cols = load_and_scale()
    # st.write("### Scaled DataFrame Preview", df_scaled.head(20))
    # # or for the full thing:
    # st.dataframe(df_scaled)

    
    G = simulate_graph(df)

    # 3. Button to run GA
    if st.button("Run Genetic Algorithm"):
        st.info("Running GA… this can take a few seconds.")
        toolbox = setup_deap(df, G, num_classes=K, weights=(w_acad, w_fri, w_neg))
        best_ind, log = run_ga(toolbox, pop_size=pop, gens=gens)

        # Decode and show results
        assignment = pd.DataFrame({
            "Student_ID": df["Student_ID"],
            "Assigned_Class": best_ind
        })
        st.success("✅ GA complete!")
        st.dataframe(assignment)

        # (Optional) visualize class-average academic scores
        # class_avgs = assignment.merge(
        #     df[["Student_ID","Academic_Composite"]],
        #     on="Student_ID"
        # ).groupby("Assigned_Class")["Academic_Composite"].mean()

        # # after you compute `assignment` and `class_avgs`
        # tabs = st.tabs(["📝 Assignments", "📊 Academics", "💗 Wellbeing", "🌐 Network"])
        
        # with tabs[0]:
        #     st.subheader("Student → Class Assignment")
        #     st.dataframe(assignment)
        #     csv = assignment.to_csv(index=False).encode()
        #     st.download_button("Download assignments CSV", csv, "allocations.csv")
        
        # with tabs[1]:
        #     st.subheader("Academic Composite by Class")
        #     st.bar_chart(class_avgs)
        
        # with tabs[2]:
        #     st.subheader("Wellbeing Composite by Class")
        #     # compute wellbeing composite the same way
        #     wb_cols = wellbeing_cols  # from data_prep
        #     df_wb = df[["Student_ID"] + wb_cols].copy()
        #     df_wb["Wellbeing_Composite"] = df_wb[wb_cols].sum(axis=1)
        #     wb_avgs = (
        #         assignment
        #         .merge(df_wb, on="Student_ID")
        #         .groupby("Assigned_Class")["Wellbeing_Composite"]
        #         .mean()
        #     )
        #     st.bar_chart(wb_avgs)
        
        # with tabs[3]:
        #     st.subheader("Social Graph by Class")
            # see next section for network visualization

        
        # st.bar_chart(class_avgs)

main()

