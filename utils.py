def add_sidebar_logo():
    st.markdown(
        """
        <style>
        .aetion-sidebar-logo {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            margin-top: -10px;
            margin-bottom: 25px;
            padding-left: 5px;
        }

        .aetion-sidebar-logo .aeti {
            color: #B8860B;
        }

        .aetion-sidebar-logo .on {
            color: #202F57;
        }
        </style>

        <div class="aetion-sidebar-logo">
            <span class="aeti"><i>Aeti</i></span><span class="on">ON</span>
        </div>
        """,
        unsafe_allow_html=True
    )
