# -*- coding: utf-8 -*-
"""
多店铺退款率预测模型训练脚本
支持合并多个店铺的订单明细数据训练统一模型
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import os, pickle, warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
MODEL_DIR = r'F:\数据模型\预测模型'
MODEL_FILE = os.path.join(MODEL_DIR, 'refund_model.pkl')
PRODUCT_RATES_FILE = os.path.join(MODEL_DIR, 'product_refund_rates.pkl')

# 训练数据文件列表（可添加更多店铺）
TRAIN_FILES = [
    r'F:\数据模型\DY43白剑虹XIUSS（王莉）艾甜甜\2026_1_18——2026_7_15订单明细.xlsx',
    r'F:\数据模型\DY27白剑虹XIUSS（邓绪伟）若安晴女装旗舰店\2026-1-21——2026-7-15明细（订单商品）.xlsx',
]
# ================================================

FEATURE_COLS = [
    'sales', 'orders', 'avg_price', 'unique_products',
    'weekday', 'day', 'is_weekend', 'is_618',
    'days_since_618_end', 'post_618', 'post_618_days',
    'refund_ma3', 'refund_ma5', 'refund_ma7', 'refund_ma14',
    'orders_ma3', 'orders_ma7', 'orders_ma14',
    'sales_ma3', 'sales_ma7', 'sales_ma14',
    'refund_lag1', 'refund_lag2', 'refund_lag3', 'refund_lag5',
    'sales_lag1', 'sales_lag2', 'sales_lag3',
    'refund_momentum1', 'refund_momentum3', 'refund_accel',
    'refund_std7', 'refund_trend',
    'sales_ratio', 'refund_declining',
    'product_mix_refund',
    'wd_0', 'wd_1', 'wd_2', 'wd_3', 'wd_4', 'wd_5', 'wd_6',
]


def load_historical(path):
    df = pd.read_excel(path)
    print(f"  Loaded {len(df)} orders, {len(df.columns)} columns")
    df['order_date'] = pd.to_datetime(df['订单日期'], errors='coerce')
    df['sales_amt'] = pd.to_numeric(df['销售金额'], errors='coerce').fillna(0)
    df['return_amt'] = pd.to_numeric(df['退货金额'], errors='coerce').fillna(0)
    df['actual_return_amt'] = pd.to_numeric(df['实退金额'], errors='coerce').fillna(0)
    df['cost'] = pd.to_numeric(df['销售成本'], errors='coerce').fillna(0)
    df['product'] = df['商品简称'].fillna('unknown')
    df['is_returned'] = df['售后分类'].astype(str).isin(
        ['普通退货', '仅退款', '换货', '拒收退货']
    ).astype(int)
    df = df.dropna(subset=['order_date'])
    df['date'] = df['order_date'].dt.date
    return df


def aggregate_daily(df):
    daily = df.groupby('date').agg(
        orders=('sales_amt', 'count'),
        sales=('sales_amt', 'sum'),
        returns=('return_amt', 'sum'),
        cost=('cost', 'sum'),
        returned_orders=('is_returned', 'sum'),
        avg_price=('sales_amt', 'mean'),
        unique_products=('product', 'nunique'),
    ).reset_index()
    daily['refund_rate'] = np.where(
        daily['sales'] > 0, daily['returns'] / daily['sales'], 0
    )
    daily['date'] = pd.to_datetime(daily['date'])
    return daily.sort_values('date').reset_index(drop=True)


def calc_product_refund_rates(df):
    prod = df.groupby(['date', 'product']).agg(
        sales=('sales_amt', 'sum'),
        returns=('return_amt', 'sum'),
    ).reset_index()
    prod['refund_rate'] = np.where(prod['sales'] > 0, prod['returns'] / prod['sales'], 0)
    return prod.groupby('product')['refund_rate'].mean().to_dict()


def aggregate_product_daily(df):
    pd2 = df.groupby(['date', 'product']).agg(
        orders=('sales_amt', 'count'),
        sales=('sales_amt', 'sum'),
        returns=('return_amt', 'sum'),
    ).reset_index()
    pd2['refund_rate'] = np.where(pd2['sales'] > 0, pd2['returns'] / pd2['sales'], 0)
    pd2['date'] = pd.to_datetime(pd2['date'])
    return pd2.sort_values(['product', 'date']).reset_index(drop=True)


def calc_product_mix(product_daily, daily):
    pw = product_daily.copy()
    pw['weighted'] = pw['refund_rate'] * pw['sales']
    pw_sum = pw.groupby('date').agg(wsum=('weighted', 'sum'), ts=('sales', 'sum')).reset_index()
    pw_sum['product_mix_refund'] = np.where(pw_sum['ts'] > 0, pw_sum['wsum'] / pw_sum['ts'], 0)
    pw_sum['date'] = pd.to_datetime(pw_sum['date'])
    daily = daily.merge(pw_sum[['date', 'product_mix_refund']], on='date', how='left')
    daily['product_mix_refund'] = daily['product_mix_refund'].fillna(0)
    return daily


def add_features(daily):
    daily['weekday'] = daily['date'].dt.weekday
    daily['day'] = daily['date'].dt.day
    daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)
    daily['is_618'] = ((daily['date'].dt.month == 6) & (daily['day'] <= 20)).astype(int)
    daily['days_since_618_end'] = (daily['date'] - pd.Timestamp('2026-06-21')).dt.days
    daily['post_618'] = (daily['days_since_618_end'] > 0).astype(int)
    daily['post_618_days'] = np.maximum(daily['days_since_618_end'], 0)

    for w in [3, 5, 7, 14]:
        daily[f'refund_ma{w}'] = daily['refund_rate'].rolling(w, min_periods=1).mean()
        daily[f'orders_ma{w}'] = daily['orders'].rolling(w, min_periods=1).mean()
        daily[f'sales_ma{w}'] = daily['sales'].rolling(w, min_periods=1).mean()

    for lag in [1, 2, 3, 5]:
        daily[f'refund_lag{lag}'] = daily['refund_rate'].shift(lag)
        daily[f'sales_lag{lag}'] = daily['sales'].shift(lag)

    daily['refund_momentum1'] = daily['refund_rate'] - daily['refund_rate'].shift(1)
    daily['refund_momentum3'] = daily['refund_rate'] - daily['refund_rate'].shift(3)
    daily['refund_accel'] = daily['refund_momentum1'] - daily['refund_momentum1'].shift(1)
    daily['refund_std7'] = daily['refund_rate'].rolling(7, min_periods=1).std().fillna(0)
    daily['refund_trend'] = daily['refund_ma3'] - daily['refund_ma7']
    daily['sales_ratio'] = np.where(
        daily['sales_ma7'] > 0, daily['sales'] / daily['sales_ma7'], 1
    )
    daily['refund_declining'] = (
            (daily['refund_rate'] < daily['refund_lag1']) &
            (daily['refund_lag1'] < daily['refund_lag2'])
    ).astype(int)

    for wd in range(7):
        daily[f'wd_{wd}'] = (daily['weekday'] == wd).astype(int)

    return daily.fillna(0)


def train():
    print("=" * 70)
    print("  多店铺合并训练模式")
    print("=" * 70)

    all_dfs = []
    for path in TRAIN_FILES:
        print(f"\n[加载] {os.path.basename(path)}")
        df = load_historical(path)
        all_dfs.append(df)
        print(f"  日期: {df['date'].min()} ~ {df['date'].max()}, 订单: {len(df)}")

    hist_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[合并] 总订单数: {len(hist_df)}")

    daily = aggregate_daily(hist_df)
    product_daily = aggregate_product_daily(hist_df)
    daily = calc_product_mix(product_daily, daily)
    prod_refund_rates = calc_product_refund_rates(hist_df)
    print(f"  商品历史退款率: {len(prod_refund_rates)} 个商品")

    daily = add_features(daily)
    daily = daily.fillna(0)

    train = daily[daily['sales'] > 0].copy()
    print(f"\n[训练] 使用 {len(train)} 天数据...")

    X_train = train[FEATURE_COLS]
    y_train = train['refund_rate']

    sw = np.ones(len(train))
    for i in range(len(train)):
        days_ago = (train['date'].max() - train.iloc[i]['date']).days
        if days_ago <= 14:
            sw[i] = 4.0
        elif days_ago <= 30:
            sw[i] = 2.5
        elif days_ago <= 60:
            sw[i] = 1.5

    gbdt = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    gbdt.fit(X_train, y_train, sample_weight=sw)
    print("  模型训练完成!")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(gbdt, f)
    print(f"\n[保存] 模型: {MODEL_FILE}")

    with open(PRODUCT_RATES_FILE, 'wb') as f:
        pickle.dump(prod_refund_rates, f)
    print(f"  商品退款率: {PRODUCT_RATES_FILE}")

    print("\n  特征重要性 (Top 10):")
    imp = sorted(
        zip(FEATURE_COLS, gbdt.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for fn, iv in imp[:10]:
        bar = '#' * int(iv * 60)
        print(f"    {fn:>25s}: {iv:.4f} {bar}")

    print("\n" + "=" * 70)
    print("  训练完成!")
    print("=" * 70)


if __name__ == '__main__':
    train()
