import streamlit as st
import asyncio

async def new():
    st.title("Octagon tester")
    st.divider()

    if st.button("Get Started"):
        st.switch_page("pages/codebase.py")


if __name__ == "__main__":
    asyncio.run(new())