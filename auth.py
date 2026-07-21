# -*- coding: utf-8 -*-
"""
登录认证模块 - 纯 Streamlit 原生实现，无第三方依赖
"""
import streamlit as st
import bcrypt
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')


@st.cache_resource
def load_auth_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8'),
    )


def check_login():
    """
    检查登录状态，返回 (None, is_logged_in, username, name)
    在页面最开头调用即可
    """
    if st.session_state.get('logged_in'):
        return None, True, st.session_state.get('username'), st.session_state.get('display_name')

    # 如果配置文件不存在，跳过登录
    if not os.path.exists(CONFIG_PATH):
        st.session_state['logged_in'] = True
        st.session_state['username'] = 'guest'
        st.session_state['display_name'] = '访客'
        return None, True, 'guest', '访客'

    config = load_auth_config()
    users = config['credentials']['usernames']

    with st.form('login_form'):
        st.markdown("### 登录")
        username_input = st.text_input('用户名')
        password_input = st.text_input('密码', type='password')
        submitted = st.form_submit_button('登录')

        if submitted:
            if username_input in users:
                user = users[username_input]
                if verify_password(password_input, user['password']):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username_input
                    st.session_state['display_name'] = user['name']
                    st.rerun()
                else:
                    st.error('用户名或密码错误')
                    st.stop()
            else:
                st.error('用户名或密码错误')
                st.stop()

    st.markdown("""
    <div style="text-align:center; padding:3rem;">
        <h2>请先登录</h2>
        <p style="color:#666;">输入用户名和密码以使用预测系统</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


def logout_button():
    """在侧边栏显示退出按钮"""
    if st.sidebar.button('退出登录'):
        for key in ['logged_in', 'username', 'display_name']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
