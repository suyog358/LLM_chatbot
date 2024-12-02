import streamlit as st

# Define the functions for different pages
def women_safety_page():
    st.title("Women Safety Department")
    st.write("This page provides details and support for women safety services.")

def emergency_services_page():
    st.title("Emergency Services")
    st.write("This page provides immediate help and support.")

def cleanliness_page():
    st.title("Cleanliness Department")
    st.write("This page helps maintain clean railway premises.")

def main():
    # Get the query parameter
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["home"])[0]

    if page == "home":
        st.title("Indian Railways Support Services")
        st.markdown("""
        <style>
        .department-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        .department-box {
            border: 2px solid #0066cc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .department-box:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="department-grid">
            <div class="department-box">
                <h2>👩 Women Safety</h2>
                <p>Ensuring safe travel for women</p>
                <a href="?page=women_safety"><button>Access Department</button></a>
            </div>
            <div class="department-box">
                <h2>🚨 Emergency Services</h2>
                <p>Immediate help and support</p>
                <a href="?page=emergency_services"><button>Access Department</button></a>
            </div>
            <div class="department-box">
                <h2>🧹 Cleanliness</h2>
                <p>Maintaining clean railway premises</p>
                <a href="?page=cleanliness"><button>Access Department</button></a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif page == "women_safety":
        women_safety_page()
    elif page == "emergency_services":
        emergency_services_page()
    elif page == "cleanliness":
        cleanliness_page()

main()
