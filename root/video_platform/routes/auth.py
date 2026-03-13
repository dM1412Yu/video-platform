# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import sqlite3
import hashlib
import uuid
from pathlib import Path

# 初始化蓝图
auth_bp = Blueprint('auth', __name__)

# 数据库路径（和项目统一）
DB_PATH = Path(__file__).parent.parent / "data" / "users.db"
# 确保数据库目录存在
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

# 密码加密（统一算法，注册/登录一致）
def encrypt_password(password, salt=None):
    if salt is None:
        salt = uuid.uuid4().hex[:8]  # 生成8位随机盐
    hash_obj = hashlib.sha256((salt + password).encode('utf-8'))
    return salt + hash_obj.hexdigest()

# 验证密码
def verify_password(plain_pwd, encrypted_pwd):
    salt = encrypted_pwd[:8]  # 提取加密时的盐
    return encrypt_password(plain_pwd, salt) == encrypted_pwd

# 获取数据库连接（统一方法）
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 支持按字段名取值
    return conn

# 登录页面 - 核心修复：添加Session设置+正确重定向
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    # 如果已登录，直接跳首页
    if 'user_id' in session:
        return redirect(url_for('video.index'))  # 适配项目路由（通常首页是video.index）
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # 空值校验
        if not username or not password:
            flash('用户名和密码不能为空！', 'error')
            return render_template('login.html')
        
        # 查询用户
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, password FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        # 验证用户和密码
        if not user:
            flash('用户名不存在！', 'error')
            return render_template('login.html')
        if not verify_password(password, user['password']):
            flash('密码错误！', 'error')
            return render_template('login.html')
        
        # 登录成功：设置Session（核心！之前缺失的关键步骤）
        session['user_id'] = user['id']
        session['username'] = user['username']
        session.permanent = True  # Session持久化（默认31天）
        flash(f'欢迎回来，{username}！', 'success')
        return redirect(url_for('video.index'))  # 跳转到视频首页（替换为你项目的首页路由）
    
    # GET请求：显示登录页
    return render_template('login.html')

# 注册页面 - 核心修复：统一加密+重定向
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_pwd = request.form.get('confirm_password', '').strip()
        
        # 基础校验
        if not username or not password:
            flash('用户名和密码不能为空！', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('密码长度不能少于6位！', 'error')
            return render_template('register.html')
        if password != confirm_pwd:
            flash('两次输入的密码不一致！', 'error')
            return render_template('register.html')
        
        # 检查用户名是否已存在
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            flash('用户名已被注册！', 'error')
            return render_template('register.html')
        
        # 加密密码并保存
        encrypted_pwd = encrypt_password(password)
        cursor.execute(
            'INSERT INTO users (username, password) VALUES (?, ?)',
            (username, encrypted_pwd)
        )
        conn.commit()
        conn.close()
        
        flash('注册成功！请登录', 'success')
        return redirect(url_for('auth.login'))  # 跳回登录页
    
    # GET请求：显示注册页
    return render_template('register.html')

# 退出登录 - 清空Session
@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('已成功退出登录', 'success')
    return redirect(url_for('auth.login'))
