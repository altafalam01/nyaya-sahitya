"""Streamlit chat UI.  Run: streamlit run ui/app.py   (the API must be running: uvicorn api.app:app)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `config`, `ui` importable under `streamlit run`

import streamlit as st

from config import settings
from ui.client import ApiError, Reply, ask
from ui.strings import LANG_CHOICES, t

st.set_page_config(page_title="Nyaya Sahitya", page_icon=":material/balance:", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user", "content": str} | {"role": "assistant", "reply": Reply}]

with st.sidebar:
    choice = st.segmented_control("Language / भाषा", list(LANG_CHOICES), default="auto",
                                  format_func=LANG_CHOICES.get, key="lang_choice") or "auto"
    if st.button(t(choice, "clear"), icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

ui_lang = choice if choice in ("en", "hi") else (st.session_state.get("last_lang") or "en")

st.title(t(ui_lang, "title"))
st.caption(t(ui_lang, "subtitle"))
st.info(t(ui_lang, "disclaimer"), icon=":material/info:")


def render_reply(reply: Reply, lang: str) -> None:
    if reply.grounded:
        st.markdown(reply.answer)
    else:
        st.warning(reply.answer, title=t(lang, "not_grounded"), icon=":material/search_off:")
    if reply.sources:
        with st.expander(f"{t(lang, 'sources')} ({len(reply.sources)})", icon=":material/menu_book:"):
            for s in reply.sources:
                flag = f" ({t(lang, 'repealed')})" if s.legal_status and s.legal_status != "in_force" else ""
                st.markdown(f"**{s.cite}**{flag}")
                st.caption(" ".join(s.text.split())[:600])
    st.caption(reply.disclaimer)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_reply(m["reply"], m["reply"].lang)

if prompt := st.chat_input(t(ui_lang, "placeholder"), max_chars=1000):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner(t(ui_lang, "thinking")):
                reply = ask(settings.api_url, prompt, lang=choice if choice in ("en", "hi") else None)
        except ApiError as e:
            st.error(t(ui_lang, f"err_{e.kind}"))
        else:
            st.session_state.last_lang = reply.lang
            st.session_state.messages.append({"role": "assistant", "reply": reply})
            render_reply(reply, reply.lang)
